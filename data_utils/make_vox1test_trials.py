import sys 

txt = sys.argv[1]
feats_scp = sys.argv[2]
ark_trials = sys.argv[3]

dic = {}
for line in open(feats_scp).readlines():
    utt, ark = line.strip().split()
    dic[utt] = ark

with open(ark_trials, 'w') as wf:
    for line in open(txt).readlines():
        gt, utt1, utt2 = line.strip().split()
        utt1 = utt1.replace('/', '-')[:-4]
        utt2 = utt2.replace('/', '-')[:-4]
        wf.write('{} {} {}\n'.format(gt, dic[utt1], dic[utt2]))
