import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random

from configs.paths import project_path
from configs.wandb_utils import init_wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="NIST_gears")
parser.add_argument('--split', default='1')
parser.add_argument('--f_type', default='BRP_features')

args = parser.parse_args()
 
num_epochs = 120

lr = 0.0005
num_layers = 10
num_f_maps = 64
if args.f_type == "I3D_features":
    _feat_dir = f"{project_path}/data/{args.dataset}/{args.f_type}/"
    _first = next((f for f in os.listdir(_feat_dir) if f.endswith('.npy')), None)
    features_dim = int(np.load(os.path.join(_feat_dir, _first)).shape[-1]) if _first else 2048
elif args.f_type == "BRP_features":
    features_dim = 768
elif "M2R2" in args.f_type:
    features_dim = 512
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = 0.5
    
if args.dataset == 'breakfast':
    lr = 0.0001

# Downsample annotation when I3D features were extracted at 30 fps but
# annotations are at 15 fps.  MockDataset features are already at 15 fps.
_no_downsample_datasets = {"50salads", "MockDataset"}
if args.dataset not in _no_downsample_datasets and args.f_type == "I3D_features":
    downsample = True
else:
    downsample = False


vid_list_file = f"{project_path}/data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = f"{project_path}/data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = f"{project_path}/data/"+args.dataset+f"/{args.f_type}/"
gt_path = f"{project_path}/data/"+args.dataset+"/annotations/"
 
mapping_file = f"{project_path}/data/"+args.dataset+"/mapping.txt"

model_dir = f"{project_path}/checkpoints/ASFormer/"+f"{args.dataset}_{args.f_type}"+"/split_"+args.split
results_dir = f"{project_path}/results/ASFormer/"+f"{args.dataset}_{args.f_type}"+"/split_"+args.split

# print()

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
 
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')
file_ptr.close()
actions_dict = dict()
for a in actions:
    if a.strip():
        actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)

# import pdb; pdb.set_trace()

trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, downsample)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, downsample)
    batch_gen_tst.read_data(vid_list_file_tst)

    run = init_wandb("ASFormer", args.dataset, args.split, args.f_type,
                     extra_config=dict(num_layers=num_layers, num_f_maps=num_f_maps,
                                       features_dim=features_dim, num_classes=num_classes,
                                       lr=lr, num_epochs=num_epochs,
                                       channel_mask_rate=channel_mask_rate))
    eval_data = (features_path, vid_list_file_tst, gt_path, actions_dict, sample_rate)
    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst, wandb_run=run, eval_data=eval_data)
    run.finish()

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, downsample)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

