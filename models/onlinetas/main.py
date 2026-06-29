import sys
import os

# Allow importing from project root (for configs.paths)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import argparse
import random
import numpy as np

import wandb
from configs.wandb_utils import init_wandb

from model import Trainer
from batch_gen import BatchGenerator
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
parser.add_argument('--dataset', default='REASSEMBLEmm')
parser.add_argument('--split', default='1')
parser.add_argument('--f_type', default='UM2R_features_bs10_onlyprio_ext_sched')
parser.add_argument('--inference_mode', default='semi_online',
                    choices=['semi_online', 'online'])
parser.add_argument('--num_epochs', type=int, default=50)
args = parser.parse_args()

# ---- Feature dimension ----
if args.f_type == 'I3D_features':
    _feat_dir = f"{project_path}/data/{args.dataset}/{args.f_type}/"
    _first = next((f for f in os.listdir(_feat_dir) if f.endswith('.npy')), None)
    features_dim = int(np.load(os.path.join(_feat_dir, _first)).shape[-1]) if _first else 2048
elif args.f_type == 'BRP_features':
    features_dim = 768
elif 'M2R2' in args.f_type or 'UM2R' in args.f_type:
    features_dim = 512
elif 'BRPOAF' in args.f_type:
    features_dim = 1536
else:
    # Try to infer from first feature file
    feat_dir = f"{project_path}/data/{args.dataset}/{args.f_type}/"
    first_file = next(
        (f for f in os.listdir(feat_dir) if f.endswith('.npy')), None
    )
    if first_file:
        features_dim = np.load(os.path.join(feat_dir, first_file)).shape[-1]
        print(f"Inferred feature dim: {features_dim}")
    else:
        raise ValueError(f"Cannot determine feature dim for f_type={args.f_type}")

# ---- Hyperparameters ----
num_layers = 10
num_f_maps = 128
w = 128          # clip size (64 for GTEA)
num_iterations = 3
num_td_layers = 3
td_heads = 8
sa_heads = 4
lr = 5e-4
num_epochs = args.num_epochs
sample_rate = 1
sigma = 1.0 / 32

# Dataset-specific confidence threshold (from paper Table 11)
theta_map = {'gtea': 0.6, 'GTEA': 0.6, '50Salads': 0.9, 'Breakfast': 0.8}
theta = theta_map.get(args.dataset, 0.9)

if args.dataset == '50Salads':
    sample_rate = 2
    w = 128

# ---- Paths ----
vid_list_file = f"{project_path}/data/{args.dataset}/splits/hand_train.split{args.split}.bundle"
vid_list_file_tst = f"{project_path}/data/{args.dataset}/splits/hand_test.split{args.split}.bundle"
features_path = f"{project_path}/data/{args.dataset}/{args.f_type}/"
gt_path = f"{project_path}/data/{args.dataset}/annotations/"
mapping_file = f"{project_path}/data/{args.dataset}/mapping.txt"

model_dir = f"{project_path}/checkpoints/onlinetas/{args.dataset}_{args.f_type}/split_{args.split}"
results_dir = f"{project_path}/results/onlinetas/{args.dataset}_{args.f_type}/split_{args.split}"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ---- Action mapping ----
with open(mapping_file, 'r') as f:
    actions = f.read().split('\n')
actions_dict = {}
for a in actions:
    parts = a.split()
    if len(parts) >= 2:
        actions_dict[parts[1]] = int(parts[0])

num_classes = len(actions_dict)

# ---- Compute T_max from training set (for post-processing) ----
def compute_T_max(vid_list_file, features_path, sample_rate):
    with open(vid_list_file, 'r') as f:
        vids = f.read().split('\n')[:-1]
    T_max = 0
    for vid in vids:
        fp = features_path + vid.split('.')[0] + '.npy'
        if os.path.exists(fp):
            T = np.load(fp).shape[0]  # (T, D) format
            T_sampled = (T + sample_rate - 1) // sample_rate
            T_max = max(T_max, T_sampled)
    return T_max

# ---- Trainer ----
trainer = Trainer(
    input_dim=features_dim,
    num_classes=num_classes,
    num_layers=num_layers,
    num_f_maps=num_f_maps,
    w=w,
    num_iterations=num_iterations,
    num_td_layers=num_td_layers,
    td_heads=td_heads,
    sa_heads=sa_heads,
)

if args.action == 'train':
    run = init_wandb("onlinetas", args.dataset, args.split, args.f_type,
                     extra_config=dict(features_dim=features_dim, num_classes=num_classes,
                                       num_layers=num_layers, num_f_maps=num_f_maps,
                                       clip_size=w, num_iterations=num_iterations,
                                       num_td_layers=num_td_layers, td_heads=td_heads,
                                       sa_heads=sa_heads, learning_rate=lr,
                                       num_epochs=num_epochs, sample_rate=sample_rate,
                                       sigma=sigma, theta=theta,
                                       inference_mode=args.inference_mode))
    batch_gen = BatchGenerator(
        num_classes, actions_dict, gt_path, features_path, sample_rate
    )
    batch_gen.read_data(vid_list_file)
    eval_data = (features_path, vid_list_file_tst, gt_path, actions_dict, sample_rate)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs,
                  batch_size=1, learning_rate=lr, device=device, wandb_run=run, eval_data=eval_data)
    run.finish()

elif args.action == 'predict':
    T_max = compute_T_max(vid_list_file, features_path, sample_rate)
    for epoch in range(1, num_epochs + 1):
        epoch_results_dir = results_dir + f"/epoch_{epoch}"
        os.makedirs(epoch_results_dir, exist_ok=True)
        trainer.predict(
            model_dir=model_dir,
            results_dir=epoch_results_dir,
            features_path=features_path,
            vid_list_file=vid_list_file_tst,
            epoch=epoch,
            actions_dict=actions_dict,
            device=device,
            sample_rate=sample_rate,
            inference_mode=args.inference_mode,
            theta=theta,
            sigma=sigma,
            T_max=T_max,
        )
