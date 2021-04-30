# coding: utf-8
# TODO: 生成test.txt, 存放在sys.argv[1]目录下
#       local/make_my_testset.pl 可以直接调用
import glob  
import random
import os, sys
from tqdm import tqdm

dataset = sys.argv[1]

f = open('{}/test.txt'.format(dataset), 'w')
wav_list = glob.glob('{}/*/*.wav'.format(dataset))
wav_list.sort()
for wav in tqdm(wav_list):
	pos_wavs = glob.glob('{}/{}/*.wav'.format(dataset, wav.split('/')[-2]))
	pos_wavs.remove(wav)

	neg_wavss = glob.glob('{}/*/*.wav'.format(dataset))
	neg_wavs = []
	for w in neg_wavss:
		if w.split(os.sep)[-2] != wav.split(os.sep)[-2]:
			neg_wavs.append(w)	

	for i in range(2):
		pos_wav = random.choice(pos_wavs)		
		neg_wav = random.choice(neg_wavs)

		f.write('1 {} {}\n'.format(wav[len(dataset)+1:],pos_wav[len(dataset)+1:]))
		f.write('0 {} {}\n'.format(wav[len(dataset)+1:],neg_wav[len(dataset)+1:]))
		pos_wavs.remove(pos_wav)
		neg_wavs.remove(neg_wav)
f.close()

# 检查生成的正负样本对是否正确
i = 0
f = open('{}/test.txt'.format(dataset), 'r')
for l in f.readlines():
	line = l.split(' ')
	if line[0] == '0':
		if line[1].split(os.sep)[0] == line[2].split(os.sep)[0]:
			i += 1
	if line[0] == '1':
		if line[1].split(os.sep)[0] != line[2].split(os.sep)[0]:
			i += 1
print(i)

"""
# 删除指定的文件
i = 0
n = 0
f = open('test.txt', 'r')
ff = open('test_ex123.txt', 'w')
for l in f.readlines():
	n += 1
	if 'W001' not in l and 'W002' not in l and 'W003' not in l:
		i += 1
		ff.write(l)
f.close()
ff.close()
print(i)
print(n)
"""

