import sys

f = open(sys.argv[1], "r")
f_new = open(sys.argv[2], "w")
for line in f.readlines():
	l, utt1, utt2 = line.strip().split(" ")
	utt1_new = utt1[::-1].replace("/", "-", 1)[::-1]
	utt2_new = utt2[::-1].replace("/", "-", 1)[::-1]
	f_new.write("{} {} {}\n".format(l, utt1_new, utt2_new))

f.close()
f_new.close()