# coding: utf-8
"""
TODO: 
        Convert m4a files to wav files.
usage: 
        python m4a_to_wav_MJ.py <m4a_path> <output_path>
        e.g. python m4a_to_wav_MJ.py /home/data/Speech_datasets/VoxCeleb2 \
                                     /home/data/Speech_datasets/VoxCeleb2_wav
"""
import os 
import sys 
import glob 
from multiprocessing import Pool
from tqdm import tqdm

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def worker(njob, m4a_list, p1, p2, output_path):
    pbar = tqdm(range(p1, p2))
    for index in pbar:
        pbar.set_description('process:{}'.format(os.getpid()))
        m4a_file = m4a_list[index]
        m4a_info = m4a_file.split('/')
        data_type = m4a_info[-5]
        spkid = m4a_info[-3]
        environment = m4a_info[-2]
        m4a_utt = m4a_info[-1]

        wav_path = os.path.join(output_path, data_type, spkid)
        create_folder(wav_path)
        wav_file = os.path.join(wav_path, environment + '_' + m4a_utt.replace('.m4a', '.wav'))

        cmd = "ffmpeg -v 8 -i {} -f wav -acodec pcm_s16le {}".format(m4a_file, wav_file)
        
        if not os.path.exists(wav_file):
            os.system(cmd)


def main():
    m4a_path = sys.argv[1]
    output_path = sys.argv[2]
    m4a_list = glob.glob(m4a_path + '/*/aac/id*/*/*.m4a')
    num_line = len(m4a_list)
    workers = 40
    print(len(m4a_list))

    p = Pool(workers)
    for njob in range(workers):
        p1 = num_line * njob // workers
        p2 = num_line * (njob+1) // workers
        p.apply_async(worker, args=[njob, m4a_list, p1, p2, output_path])
    p.close()
    p.join()

if __name__ == '__main__':
    main()



