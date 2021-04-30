import matplotlib.pyplot as plt 
import glob
import sys 
import os 

data_path = sys.argv[1]

spk2numUtt_dict = {}
for spk in glob.glob(data_path+'/*'):
    spk2numUtt_dict[spk] = len(os.listdir(spk))

spk2numUtt_sorted = sorted(spk2numUtt_dict.items(), key= lambda x: x[1])

spk2num = [item[1] for item in spk2numUtt_sorted ]

plt.plot(range(len(spk2num)), spk2num)
plt.xlabel('spkid')
plt.ylabel('utts (total {})'.format(str(sum(spk2num))))
plt.show()
