# coding: utf-8
"""
TODO: 
        Use the sliding window to cut the long audio file into fixed-length fragments
usage: 
        python cut_wav_to_fixLenSeg.py <indata_path> <outdata_path>
        e.g. python cut_wav_to_fixLenSeg.py /home/data/Speech_datasets/VoxCeleb2_wav/dev \
                                            /home/data/Speech_datasets/VoxCeleb2_wav/dev_seg
"""
import sys 
import soundfile as sf 
import os 
from tqdm import tqdm
import glob 
import shutil
import numpy as np 
from multiprocessing import Pool


def worker(njob, wav_list, p1, p2, output_path, win_len, win_shift):
    pbar = tqdm(range(p1, p2))
    for index in pbar:
        pbar.set_description('process:{}'.format(os.getpid()))
        wav = wav_list[index]
        path, utt = os.path.split(wav)
        environment = path.split('/')[-1]
        spkid = path.split('/')[-2]
        utt = environment + "_" + utt
        dst = os.path.join(output_path, spkid)
        
        if len(glob.glob(os.path.join(dst, utt).replace('.wav', '*')))>0:
            continue

        if not os.path.exists(dst):
            os.makedirs(dst)

        audio, sr = sf.read(wav)
        num_samples = audio.shape[0]
        # 小于win_len + win_shift时不做切分
        if num_samples <= (win_len + win_shift)*sr:
            dst_wav = os.path.join(dst, utt)
            sf.write(dst_wav, audio, sr)

        # 大于win_len + win_shift时，切成至少win_len每段，重叠win_len - win_shift
        else:
            audio_segs = []
            for start in range(0, num_samples, int(win_shift*sr)):
                end = start + int(win_len*sr)
                if end < num_samples:
                    #if (num_samples - start) <= win_shift*sr:
                    seg = audio[start: end]
                    audio_segs.append(seg)
                else:
                    seg = audio[start-int((win_shift-(win_len-win_shift))*sr) : ]
                    audio_segs[-1] = seg
                    break 

            for i, segment in enumerate(audio_segs):
                utt_ = utt.replace('.wav', '_{}.wav'.format(str(i)))
                dst_wav = os.path.join(dst, utt_)
                sf.write(dst_wav, segment, sr)

def main():

    params = sys.argv[1:]
    input_path = params[0]
    output_path = params[1]
    #input_path = "/home/data/Speech_datasets/VoxCeleb2_wav/dev"
    #output_path = "/home/data/Speech_datasets/VoxCeleb2_wav/dev_seg"
    win_len = 5
    win_shift = 4

    njob = 42
    #wav_list = glob.glob(input_path + "/*/*.wav")
    wav_list = []
    for spk in glob.glob(input_path+'/*'):
        if not os.path.exists(os.path.join(output_path, spk.split('/')[-1])):
            wav_list += glob.glob(spk+'/*/*.wav')

    num_wavs = len(wav_list)
    print('Total numbers of wavs is:', num_wavs)

    #worker(0, wav_list, 0, 1000, output_path, win_len, win_shift); a += 1

    p = Pool(njob)
    for nj in range(njob):
        p1 = num_wavs * nj // njob
        p2 = num_wavs * (nj+1) // njob
        p.apply_async(worker, args=[nj, wav_list, p1, p2, output_path, win_len, win_shift])
    p.close()
    p.join()


if __name__ == "__main__":
    main()