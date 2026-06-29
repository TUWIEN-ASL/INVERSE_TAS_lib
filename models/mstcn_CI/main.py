#!/usr/bin/python2.7

import torch
from models.mstcn_CI.model_def import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import numpy as np
import glob

from configs.paths import project_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="NIST_gears")
parser.add_argument('--split', default='0')
parser.add_argument('--f_type', default='BRP_features')
parser.add_argument('--use_const', action="store_true")
parser.add_argument('--predict_mode', default='last', choices=['all', 'last'], 
                   help='Generate predictions for all checkpoints or just the last one')

args = parser.parse_args()

num_stages = 1
num_layers = 5
num_f_maps = 64
if args.f_type == "I3D_features":
    features_dim = 2048
elif args.f_type == "BRP_features":
    features_dim = 768
elif "M2R2" in args.f_type:
    features_dim = 512

if args.use_const:
    N = 30
    D = 12
    const_dim = N*D
else:
    const_dim = 0

bz = 1
lr = 0.0005
num_epochs = 50

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

if args.dataset != "50salads" and args.f_type == "I3D_features":
    downsample = True
else:
    downsample = False

vid_list_file = f"{project_path}/data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = f"{project_path}/data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = f"{project_path}/data/"+args.dataset+f"/{args.f_type}/"
gt_path = f"{project_path}/data/"+args.dataset+"/annotations/"
prio_path = f"{project_path}/data/"+args.dataset+"/prio/"

mapping_file = f"{project_path}/data/"+args.dataset+"/mapping.txt"

model_dir = f"{project_path}/checkpoints/mstcn/"+f"{args.dataset}_{args.f_type}"+"/split_"+args.split
results_dir = f"{project_path}/results/mstcn/"+f"{args.dataset}_{args.f_type}"+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, const_dim, num_classes)

def get_available_checkpoints(model_dir):
    """
    Get all available checkpoint files in the model directory.
    
    Args:
        model_dir: Directory containing model checkpoints
        
    Returns:
        List of tuples (epoch_number, checkpoint_path)
    """
    checkpoint_pattern = os.path.join(model_dir, "epoch-*.model")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    checkpoints = []
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        # Extract epoch number from filename like "epoch-50.model"
        epoch_str = filename.split('-')[1].split('.')[0]
        try:
            epoch_num = int(epoch_str)
            checkpoints.append((epoch_num, file_path))
        except ValueError:
            continue
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, prio_path, sample_rate, downsample)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    # Get available checkpoints
    checkpoints = get_available_checkpoints(model_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {model_dir}")
        exit(1)

    print(f"Found {len(checkpoints)} checkpoints")
    
    # Determine which checkpoints to use for prediction
    if args.predict_mode == 'last':
        # Use only the last checkpoint
        checkpoints_to_predict = [checkpoints[-1]]
        print(f"Generating predictions for last checkpoint: epoch {checkpoints[-1][0]}")
    else:
        # Use all checkpoints
        checkpoints_to_predict = checkpoints
        print(f"Generating predictions for all {len(checkpoints)} checkpoints")

    for epoch_num, checkpoint_path in checkpoints_to_predict:
        print(f"\nGenerating predictions for epoch {epoch_num}...")
        
        # Create epoch-specific results directory if evaluating all checkpoints
        if args.predict_mode == 'all':
            epoch_results_dir = f"{results_dir}/epoch_{epoch_num}"
            if not os.path.exists(epoch_results_dir):
                os.makedirs(epoch_results_dir)
        else:
            epoch_results_dir = results_dir
        
        try:
            trainer.predict(model_dir, epoch_results_dir, features_path, prio_path, 
                          vid_list_file_tst, epoch_num, actions_dict, device, sample_rate)
            print(f"Predictions saved to {epoch_results_dir}")
        except Exception as e:
            print(f"Error generating predictions for epoch {epoch_num}: {str(e)}")
            continue
    
    print(f"\nPrediction generation complete!")
    if args.predict_mode == 'all':
        print(f"Results saved in separate directories under {results_dir}/epoch_X/")
        print(f"To evaluate all checkpoints, run:")
        print(f"python eval.py --dataset {args.dataset} --split {args.split} --f_type {args.f_type} --eval_mode all")
    else:
        print(f"Results saved in {results_dir}")
        print(f"To evaluate, run:")
        print(f"python eval.py --dataset {args.dataset} --split {args.split} --f_type {args.f_type} --eval_mode last")