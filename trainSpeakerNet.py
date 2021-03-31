#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
from time import strftime, localtime
from datetime import datetime
import yaml
import numpy
import pdb

import glob
import zipfile
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import get_data_loader
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk',type=int,   default=200,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')

## Training details
parser.add_argument('--test_interval',  type=int,   default=5,      help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=250,    help='Maximum number of epochs')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="ReduceLROnPlateau", help='Learning rate scheduler, ReduceLROnPlateau or steplr or OneCycleLR')
parser.add_argument('--lr',             type=float, default=0.005,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.5,    help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=5e-5,   help='Weight decay in the optimizer')

## Loss functions
parser.add_argument('--trainfunc',      type=str,   default="ge2e", help='Loss function')
parser.add_argument('--margin',         type=float, default=0.3,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=2,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=1211,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--alpha',          type=float, default=0.5,    help='Loss weight for *proto loss (0,1) loss = (1-r)*softmaxbase + r*proto')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="/home/data_ssd/lcf/voxceleb_trainer_exps/exp_ResNetSE34L_vox1_trainSeg_ge2e", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="/home/data_ssd/lcf/API-Net_VoxExp/data/vox1_train/feats.scp",       help='Train list')
parser.add_argument('--test_list',      type=str,   default="/home/data_ssd/lcf/API-Net_VoxExp/data/vox1_test/trials_ark",     help='Evaluation list')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="ResNetSE34L",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="ASP",  help='Type of encoder')
parser.add_argument('--MultiStageAgg',  type=bool,  default=False,  help='usage multi-stage aggregation')
parser.add_argument('--MSA_type',       type=str,   default='234',  help='MSA strategies')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--final_bn',       type=bool,  default=False,  help='usage final bn layer')
parser.add_argument('--vlad_drop',      type=float, default=0.25,   help='dropout of the VLAD')

## for VLAD
parser.add_argument('--num_clusters',   type=str,   default=10,     help='Number of clusters in VLAD')
parser.add_argument('--groups',         type=str,   default=8,      help='Number of groups in NeXtVLAD')
parser.add_argument('--expansion',      type=str,   default=2,      help='expansion in NeXtVLAD')

## For test only
parser.add_argument('--eval',           dest='eval',action='store_true', help='Eval only')
parser.add_argument('--extract_embedding', dest='extract_embedding', action='store_true', help='Extract embeddings only')
parser.add_argument('--GPUid',          type=str,   default="1", help='指定gpu id')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args()

seed = 19260817
torch.manual_seed(seed)    # set random seed for cpu
torch.cuda.manual_seed(seed)    # set random seed for current GPU
torch.cuda.manual_seed_all(seed)    # set random seed for all GPU
numpy.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            if typ is not None:
                args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPUid

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Trainer script
# ## ===== ===== ===== ===== ===== ===== ===== =====
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    it          = 1
    min_eer     = [100]

    ## Load models
    s = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port
        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)
        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)
        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)
        print('Loaded the model on GPU %d'%args.gpu)
    else:
        s = WrappedModel(s).cuda(args.gpu)

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(args.train_list, **vars(args))
    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()
    if len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model %s loaded from previous state!"%modelfiles[-1])
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    elif(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model %s loaded!"%args.initial_model)

    ## Evaluation code - must run on single GPU
    if args.eval == True:
        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())
        print('Total parameters: ',pytorch_total_params)
        print('Test list',args.test_list)
        assert args.distributed == False
        sc, lab, all_trials = trainer.evaluateFromList(**vars(args))
        result = tuneThresholdfromScore(sc, lab, [1, 0.1])

        p_target = 0.01
        c_miss = 1
        c_fa = 1

        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

        print('EER %2.4f MinDCF %.5f'%(result[1],mindcf))
        quit()

    ## Extract_embedding
    if args.extract_embedding == True:
        trainer.extract_embeds(args.test_list, print_interval=10, save_path='embedding', eval_frames=args.eval_frames,num_eval=1)
        quit()

    ## Write args to scorefile
    scorefile   = open(args.result_save_path+"/scores.txt", "a+")
    scorefile.write("="*20+strftime("%Y-%m-%d %H:%M:%S", localtime())+"="*20+'\n')
    for items in vars(args):
        print('%s: %s'%(items, vars(args)[items]))
        scorefile.write('%s: %s\n'%(items, vars(args)[items]))
    scorefile.flush()

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py') + glob.glob('./*/*.py')
        strtime = strftime("%Y%m%d%H%M%S", localtime())
        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()
        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)

    ## Core training script
    for it in range(it,args.max_epoch+1):
        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]
        
        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with %s LR %f..."%(args.model, args.trainfunc, max(clr)))

        loss, traineer = trainer.train_network(trainLoader, verbose=(args.gpu == 0))
        trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

        if it % args.test_interval == 0 and args.gpu == 0:
            ## Perform evaluation only in single GPU training
            if not args.distributed:
                sc, lab, _ = trainer.evaluateFromList(**vars(args))
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])
                min_eer.append(result[1])

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)

            print("IT %d"%it, time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f"%( max(clr), traineer, loss, result[1], min(min_eer)))
            scorefile.write(time.strftime("%Y-%m-%d %H:%M:%S")+" IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, minDCF %2.4f, VEER %2.4f, MINEER %2.4f\n"%(it, max(clr), traineer, loss, mindcf, result[1], min(min_eer)))
            scorefile.flush()

            args.writer.add_scalar('EER', result[1], it)

        elif it % args.test_interval != 0 and args.gpu == 0:
            
            scorefile.write(time.strftime("%Y-%m-%d %H:%M:%S")+" IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n"%(it, max(clr), traineer, loss))
            scorefile.flush()
    
    scorefile.write("*"*20+strftime("%Y-%m-%d %H:%M:%S", localtime())+"*"*20+'\n')
    scorefile.close()


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====
def main():

    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    if not(os.path.exists(args.model_save_path)):
        os.makedirs(args.model_save_path)
            
    if not(os.path.exists(args.result_save_path)):
        os.makedirs(args.result_save_path)

    # tensorboard writer
    ## tensorboardX writer
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, 'tensorboard_log', current_time))
    args.writer = writer

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()
