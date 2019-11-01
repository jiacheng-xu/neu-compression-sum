path_of_files = '/home/cc/final-cnn/merge/train/data_byline_beam'
target_wr_file = '/home/cc/final-cnn/merge/train/demo-train.txt'
import os

os.chdir(path_of_files)
files = os.listdir(path_of_files)

bag = []
for f in files:
    with open(f, 'r') as fd:
        lines = fd.read().splitlines()
        assert len(lines) == 1
        bag.append(lines[0])
with open(target_wr_file, 'w') as fd:
    print(len(bag))
    fd.write('\n'.join(bag))
