# root = '/backup3/jcxu'
# root = '/home/cc'
root = "/scratch/cluster/jcxu"

data = 'cnndm'

prefix = "PYTHONPATH=./ python3 neusum/train_model.py "
s = [
    " "
    " --abs_board_file {}/exComp/board.txt  --pretrain_embedding "
    " {}/data/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec "
    "  --abs_dir_log {}/log  --abs_dir_root {}/exComp   " \
    " --abs_dir_exp {}/exp "
    "  --lazy --batch_size 35 --elmo --eval_batch_size 40 ".format(root, root, root, root, root)
]
# "--compression --fix_edu_num -1 --alpha 0.5"
dbg = False
if dbg:
    s += "  --dbg"

# compression:
if data == 'dm':
    s = [compression + seed for seed in s
         for compression in [
             "  --fix_edu_num 3 --max_dec_step 3 --min_dec_step 3  --epochs 70 ",
             "  --compression --aggressive_compression -1 --fix_edu_num 3 --epochs 70 ",
             "  --compression --aggressive_compression -1 --fix_edu_num -1 --epochs 70 ",
             # lead compression

             # "  --compression --trim_sent_oracle --aggressive_compression -1  ",  # sent only with compression sent oracle
             # "  --compression  --aggressive_compression -1   ",  # joint
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num 3 --epochs 70 ",
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num 3 --alpha 10 ",
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num -1  ",
             # "  --compression  --aggressive_compression -1 --fix_edu_num  3 ",
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num 4 --max_dec_step 4 --min_dec_step 4 ",
         ]]

elif data == 'cnn':
    s = [compression + seed for seed in s
         for compression in [

             "  --fix_edu_num 3 --max_dec_step 3 --min_dec_step 3  --epochs 70 ",
             "  --compression --aggressive_compression -1 --fix_edu_num 3 --epochs 70 ",
             "  --compression --aggressive_compression -1 --fix_edu_num -1 --epochs 70 ",
             # "  --compression  --aggressive_compression -1  --compress_leadn 3 --fix_edu_num 3  ",  # lead compression
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num 3  ",
             # " --fix_edu_num 3 --max_dec_step 3 --min_dec_step 3 "
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num -1  ",
             # "  --compression  --aggressive_compression -1 --fix_edu_num  3 ",
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num 4 --max_dec_step 4 --min_dec_step 4 ",
         ]]
elif data == 'cnndm':
    # cnndm_pre = " --load_saved_model 968-data_cnndm-cmpres_True-step_3-leadn_-1-lr_0.001-dbg_False-alpha_1.0-optim_adam-schedule_1.0-avdTri_False/best.th  "
    cnndm_pre=" "
    s = [compression + seed for seed in s
         for compression in [
             "  --compression  --aggressive_compression -1  --fix_edu_num -1 --epochs 70    " + cnndm_pre,
             # lead compression
             "  --compression --aggressive_compression -1 --fix_edu_num 3 --epochs 70 " + cnndm_pre,
             " --fix_edu_num 3 --max_dec_step 3 --min_dec_step 3 " + cnndm_pre
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num -1  "
         ]]
elif data == 'nyt':
    s = [compression + seed for seed in s
         for compression in [

             "  --compression  --aggressive_compression -1  --compress_leadn 5 --fix_edu_num 5 ",  # lead compression
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num 4 --max_dec_step 4 --min_dec_step 4 ",
             "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num 5 --max_dec_step 5 --min_dec_step 5 ",
             "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num 5 --max_dec_step 5 --min_dec_step 5 --alpha 5",
             # " --fix_edu_num 5 --max_dec_step 5 --min_dec_step 5 "
             # joint
             # "  --compression --trim_sent_oracle --aggressive_compression -1 --fix_edu_num 3 --alpha 5 ",
             # "  --compression  --aggressive_compression -1  --fix_edu_num 3 --alpha 5 ",  # joint
         ]]
    #     "--load_saved_model "
    #     " /scratch/cluster/jcxu/exp/156-data_nyt-cmpres_True-step_5-leadn_-1-lr_0.001-dbg_False-alpha_5.0-optim_adam-schedule_1.0-avdTri_False/best.th  "
if data == 'dm' or data == 'cnn':
    s = [d + seed for seed in s for d in
         [
             # " --abs_dir_data {}/data/read_ready-grammarTrue-miniFalse-maxsent20-beam5".format(root),
             "  --data_name {} --abs_dir_data {}/data/2merge-cnndm-ilp --train_fname train.pkl.{}   --dev_fname dev.pkl.{}  --test_fname test.pkl.{} ".format(
                 data, root, data, data, data),
         ]
         ]
else:
    s = [d + seed for seed in s for d in
         [
             # " --abs_dir_data {}/data/read_ready-grammarTrue-miniFalse-maxsent20-beam5".format(root),
             # " --data_name dm --abs_dir_data {}/data/dm/dm-gramTrue-miniFalse-maxsent30-beam8".format(root),
             " --data_name {} --abs_dir_data {}/data/2merge-{}-ilp/ --train_fname train.pkl   --dev_fname dev.pkl  --test_fname test.pkl ".format(
                 data, root, data)
         ]
         ]

s = [seed + sch for seed in s for sch in [" --schedule 1.0 "
                                          # , " --schedule 0.2 "
                                          ]]

# s = [" --fix_edu_num " + edu_num + "  " + seed for seed in s for edu_num in ["2"]]

import random

# random.shuffle(s)
s = [prefix + x + " --device_cuda_id  " for x in s]
print("\n\n".join(s))

gpu_num = 4
# --device_cuda_id 0
total = len(s)

# print("trap '{ echo \"Hey, you pressed Ctrl-C. Press Ctrl-D or Kill this screen to kill this screen.\" ; }' INT\n")
