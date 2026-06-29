import numpy as np
import os
import argparse

# Setup argument parsing
parser = argparse.ArgumentParser(description="Generate frame splits based on root path.")
parser.add_argument('root', type=str, help='Root directory path')
parser.add_argument('split', type=int, help='split number')
args = parser.parse_args()

# Define variables
modes = ['train', 'test']
n_split = args.split
num_frames = 8
overlap = [1, 1]
dss = [24, 32]

# Compute the frame directory based on root path
frame_dir = os.path.join(args.root, 'frames')

# Process each mode
for mode in modes:
    txt_path = f'splits/{mode}.split{n_split}.bundle'
    train_split = []
    
    # Read train split file
    with open(os.path.join(args.root, txt_path), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split('.')[0]
            train_split.append(line)

    new_train_list = []
    
    # Iterate through the splits and compute frame indexes
    for i in range(len(dss)):
        for dat in train_split:
            vpath = os.path.join(frame_dir, dat)
            vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
            start_idxs = np.arange(0, vlen, int(num_frames * overlap[i] * dss[i]))
            for idx in start_idxs:
                new_train_list.append([dat, idx, dss[i]])

    # Save the new train list to a .npy file
    np.save(os.path.join(args.root, f'splits/{mode}_split{n_split}_nf{num_frames}_ol{overlap}_ds{dss}.npy'),
            np.array(new_train_list))


exfm = []
for mode in modes:
    txt_path = f'splits/{mode}.split{n_split}.bundle'

    # Read train split file
    with open(os.path.join(args.root, txt_path), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split('.')[0]
            train_split.append(line)

    for dat in train_split:
            vpath = os.path.join(frame_dir, dat)
            vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
            start_idxs = np.arange(0, vlen, 32)
            for idx in start_idxs:
                exfm.append([dat, idx])

# Save the new train list to a .npy file
np.save(os.path.join(args.root, f'splits/exfm_nf32.npy'),
        np.array(exfm))