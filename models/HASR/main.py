import argparse
import importlib.util
import os
import sys
import random
import torch
import numpy as np

# Add project root and HASR dir to sys.path (HASR first to resolve its model.py)
_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_dir, '..', '..'))
for p in [_root, _dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.paths import project_path                           # noqa: E402
from configs.wandb_utils import init_wandb                       # noqa: E402
from model import GRURefiner, TransformerHASR                   # models/HASR/model.py
from batch_gen import BatchGenerator                            # models/HASR/batch_gen.py
from train import train_refiner                                 # models/HASR/train.py
from predict import predict_refiner                             # models/HASR/predict.py
from online_train import train_online_refiner                   # models/HASR/online_train.py
from streaming_infer import run_online_inference                # models/HASR/streaming_infer.py


def _load_module(name, path):
    """Load a Python module from an absolute path without polluting sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Train / predict with HASR refiner (offline or online)'
)
parser.add_argument('--action',   default='train',
                    choices=['train', 'predict', 'online_train', 'online_infer'])
parser.add_argument('--dataset',  default='ImPerfectPour')
parser.add_argument('--split',    default='1')
parser.add_argument('--f_type',   default='BRP_features')
parser.add_argument('--backbone', default='mstcn',
                    choices=['mstcn', 'ASFormer', 'onlinetas'],
                    help='Backbone for offline (mstcn/ASFormer) or online (onlinetas) training')
parser.add_argument('--refiner',  default='transformer', choices=['transformer', 'gru'])

# Shared refiner hyperparameters
parser.add_argument('--num_epochs',     type=int,   default=60)
parser.add_argument('--lr',             type=float, default=1e-4)
parser.add_argument('--feat_dim',       type=int,   default=512)
parser.add_argument('--dropout',        type=float, default=0.1)
parser.add_argument('--num_hl_frames',  type=int,   default=32)
parser.add_argument('--num_hl_samples', type=int,   default=64)

# Transformer-specific
parser.add_argument('--d_model',            type=int, default=512)
parser.add_argument('--nhead',              type=int, default=8)
parser.add_argument('--transformer_layers', type=int, default=4)
parser.add_argument('--dim_feedforward',    type=int, default=2048)

# GRU-specific
parser.add_argument('--gru_layers', type=int, default=1)

# Online-mode args
parser.add_argument('--prefix_strategy', default='clip_boundaries',
                    choices=['random', 'clip_boundaries'],
                    help='Online training prefix-sampling strategy')
parser.add_argument('--clip_width', type=int, default=128,
                    help='OnlineTAS clip width w (default: 128)')

# Backbone epoch control
parser.add_argument('--backbone_epoch', type=int, default=0,
                    help='Backbone epoch for offline predict (0 = auto: last)')
parser.add_argument('--refiner_epoch',  type=int, default=30)

args = parser.parse_args()

# ── Reproducibility ───────────────────────────────────────────────────────────
seed = 1538574472
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Feature dimension ─────────────────────────────────────────────────────────
if args.f_type == 'I3D_features':
    _feat_dir = f"{project_path}/data/{args.dataset}/{args.f_type}/"
    _first = next((f for f in os.listdir(_feat_dir) if f.endswith('.npy')), None)
    features_dim = int(np.load(os.path.join(_feat_dir, _first)).shape[-1]) if _first else 2048
elif args.f_type == 'BRP_features':
    features_dim = 768
elif 'M2R2' in args.f_type:
    features_dim = 512
elif 'BRPOAF' in args.f_type:
    features_dim = 768 + 768
elif 'UM2R' in args.f_type:
    features_dim = 512
else:
    features_dim = 768

# ── Paths ─────────────────────────────────────────────────────────────────────
data_root      = f'{project_path}/data/{args.dataset}'
features_path  = f'{data_root}/{args.f_type}/'
gt_path        = f'{data_root}/annotations/'
mapping_file   = f'{data_root}/mapping.txt'
vid_list_train = f'{data_root}/splits/hand_train.split{args.split}.bundle'
vid_list_test  = f'{data_root}/splits/hand_test.split{args.split}.bundle'

# Online actions append '_online' to the run tag so checkpoints never collide
_online_suffix = '_online' if args.action in ('online_train', 'online_infer') else ''
_run_tag = f'{args.dataset}_{args.f_type}_{args.backbone}_{args.refiner}{_online_suffix}'
hasr_model_dir    = f'{project_path}/checkpoints/HASR/{_run_tag}/split_{args.split}'
hasr_results_base = f'{project_path}/results/HASR/{_run_tag}/split_{args.split}'

os.makedirs(hasr_model_dir, exist_ok=True)
os.makedirs(hasr_results_base, exist_ok=True)

# ── Action mapping ────────────────────────────────────────────────────────────
with open(mapping_file, 'r') as f:
    actions_raw = [line for line in f.read().split('\n') if line.strip()]

actions_dict = {}
for a in actions_raw:
    parts = a.split()
    actions_dict[parts[1]] = int(parts[0])

num_classes = len(actions_dict)
sample_rate = 2 if args.dataset == '50Salads' else 1

# ── Backbone ──────────────────────────────────────────────────────────────────
MemoryBank = None  # only used for onlinetas

if args.backbone == 'mstcn':
    _bb_mod = _load_module('mstcn_model', os.path.join(_root, 'models', 'mstcn', 'model.py'))
    backbone = _bb_mod.MultiStageModel(
        num_stages=5, num_layers=10, num_f_maps=64,
        dim=features_dim, num_classes=num_classes,
    )
    backbone_model_dir     = f'{project_path}/checkpoints/mstcn/{args.dataset}_{args.f_type}/split_{args.split}'
    backbone_epochs        = list(range(1, 51))
    backbone_predict_epoch = args.backbone_epoch if args.backbone_epoch > 0 else 50

elif args.backbone == 'ASFormer':
    _bb_mod = _load_module('asformer_model', os.path.join(_root, 'models', 'ASFormer', 'model.py'))
    backbone = _bb_mod.MyTransformer(
        num_decoders=3, num_layers=10, r1=2, r2=2,
        num_f_maps=64, input_dim=features_dim,
        num_classes=num_classes, channel_masking_rate=0.3,
    )
    backbone_model_dir     = f'{project_path}/checkpoints/ASFormer/{args.dataset}_{args.f_type}/split_{args.split}'
    backbone_epochs        = list(range(10, 121, 10))
    backbone_predict_epoch = args.backbone_epoch if args.backbone_epoch > 0 else 120

elif args.backbone == 'onlinetas':
    _ot_mod = _load_module('onlinetas_model', os.path.join(_root, 'models', 'onlinetas', 'model.py'))
    MemoryBank = _ot_mod.MemoryBank

    # Infer architecture from first available checkpoint so we never mismatch.
    _bb_ckpt_dir = f'{project_path}/checkpoints/onlinetas/{args.dataset}_{args.f_type}/split_{args.split}'
    _first_ckpt  = None
    if os.path.isdir(_bb_ckpt_dir):
        _ckpts = sorted([f for f in os.listdir(_bb_ckpt_dir) if f.endswith('.model')])
        if _ckpts:
            _first_ckpt = os.path.join(_bb_ckpt_dir, _ckpts[0])

    _num_iterations = 2   # OnlineTASModel default
    _num_td_layers  = 2
    _num_f_maps     = 128
    _num_layers     = 10
    if _first_ckpt:
        _sd = torch.load(_first_ckpt, map_location='cpu', weights_only=False)
        # Count sa_layers indices → num_iterations
        _sa_idx = {int(k.split('.')[2]) for k in _sd if k.startswith('cfa.sa_layers.')}
        if _sa_idx:
            _num_iterations = max(_sa_idx) + 1
        # Count td_layer indices within iteration 0 → num_td_layers
        _td_idx = {int(k.split('.')[3]) for k in _sd if k.startswith('cfa.td_layers.0.')}
        if _td_idx:
            _num_td_layers = max(_td_idx) + 1
        # Infer num_f_maps from tcn conv_1x1 weight
        _fm_key = next((k for k in _sd if 'tcn.conv_1x1.weight' in k), None)
        if _fm_key:
            _num_f_maps = _sd[_fm_key].shape[0]
        # Infer num_layers from tcn dilated layers
        _layer_idx = {int(k.split('.')[2]) for k in _sd if k.startswith('tcn.layers.')}
        if _layer_idx:
            _num_layers = max(_layer_idx) + 1
        print(f'OnlineTAS arch inferred from checkpoint: '
              f'num_iterations={_num_iterations}, num_td_layers={_num_td_layers}, '
              f'num_f_maps={_num_f_maps}, num_layers={_num_layers}')

    backbone = _ot_mod.OnlineTASModel(
        input_dim=features_dim, num_classes=num_classes,
        num_layers=_num_layers, num_f_maps=_num_f_maps, w=args.clip_width,
        num_iterations=_num_iterations, num_td_layers=_num_td_layers,
    )
    backbone_model_dir     = f'{project_path}/checkpoints/onlinetas/{args.dataset}_{args.f_type}/split_{args.split}'
    # OnlineTAS saves every epoch; scan what's actually there
    if os.path.isdir(backbone_model_dir):
        backbone_epochs = sorted([
            int(f.replace('epoch-', '').replace('.model', ''))
            for f in os.listdir(backbone_model_dir)
            if f.startswith('epoch-') and f.endswith('.model')
        ])
    else:
        backbone_epochs = []
    backbone_predict_epoch = args.backbone_epoch if args.backbone_epoch > 0 else (backbone_epochs[-1] if backbone_epochs else 50)

backbone.to(device)

# ── Refiner ───────────────────────────────────────────────────────────────────
_shared = dict(
    num_actions=num_classes,
    input_dim=features_dim,
    feat_dim=args.feat_dim,
    dropout=args.dropout,
    num_highlevel_frames=args.num_hl_frames,
    num_highlevel_samples=args.num_hl_samples,
    device=str(device),
)

if args.refiner == 'transformer':
    refiner = TransformerHASR(
        **_shared,
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.transformer_layers,
        dim_feedforward=args.dim_feedforward,
    )
else:
    refiner = GRURefiner(**_shared, num_gru_layers=args.gru_layers)

print(f'Backbone: {args.backbone}  |  Refiner: {args.refiner}  |  Action: {args.action}')
refiner.to(device)

# ── Dispatch ──────────────────────────────────────────────────────────────────
if args.action == 'train':
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_train)
    run = init_wandb(f"HASR_{args.backbone}_{args.refiner}", args.dataset, args.split, args.f_type,
                     extra_config=dict(backbone=args.backbone, refiner=args.refiner,
                                       num_epochs=args.num_epochs, lr=args.lr,
                                       feat_dim=args.feat_dim, num_classes=num_classes))
    eval_data = (features_path, vid_list_test, gt_path, actions_dict, sample_rate)
    train_refiner(
        model=refiner, backbone=backbone,
        backbone_model_dir=backbone_model_dir, backbone_epochs=backbone_epochs,
        batch_gen=batch_gen, num_epochs=args.num_epochs,
        learning_rate=args.lr, model_dir=hasr_model_dir, device=device,
        wandb_run=run, eval_data=eval_data,
    )
    run.finish()

elif args.action == 'predict':
    epochs_to_predict = sorted([
        int(f.replace('epoch-', '').replace('.model', ''))
        for f in os.listdir(hasr_model_dir)
        if f.startswith('epoch-') and f.endswith('.model')
    ]) if os.path.isdir(hasr_model_dir) else []

    if not epochs_to_predict:
        raise RuntimeError(f'No refiner checkpoints found in {hasr_model_dir}')

    print(f'Predicting for {len(epochs_to_predict)} refiner epochs: {epochs_to_predict}')
    for ep in epochs_to_predict:
        epoch_result_dir = os.path.join(hasr_results_base, f'epoch_{ep}')
        os.makedirs(epoch_result_dir, exist_ok=True)
        print(f'  epoch {ep} …', flush=True)
        predict_refiner(
            model=refiner, backbone=backbone,
            backbone_model_dir=backbone_model_dir, backbone_epoch=backbone_predict_epoch,
            model_dir=hasr_model_dir, refiner_epoch=ep,
            result_dir=epoch_result_dir, features_path=features_path,
            vid_list_file=vid_list_test, actions_dict=actions_dict,
            device=device, sample_rate=sample_rate,
        )

elif args.action == 'online_train':
    if args.backbone != 'onlinetas':
        raise ValueError('--action online_train requires --backbone onlinetas')
    if not backbone_epochs:
        raise RuntimeError(f'No OnlineTAS checkpoints found in {backbone_model_dir}')

    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_train)
    run = init_wandb(f"HASR_{args.backbone}_{args.refiner}_online", args.dataset, args.split, args.f_type,
                     extra_config=dict(backbone=args.backbone, refiner=args.refiner,
                                       num_epochs=args.num_epochs, lr=args.lr,
                                       prefix_strategy=args.prefix_strategy,
                                       clip_width=args.clip_width, num_classes=num_classes))
    train_online_refiner(
        model=refiner, backbone=backbone, MemoryBank=MemoryBank,
        backbone_model_dir=backbone_model_dir, backbone_epochs=backbone_epochs,
        batch_gen=batch_gen, num_epochs=args.num_epochs,
        lr=args.lr, model_dir=hasr_model_dir, device=device,
        prefix_strategy=args.prefix_strategy, clip_width=args.clip_width,
        wandb_run=run,
    )
    run.finish()

elif args.action == 'online_infer':
    if args.backbone != 'onlinetas':
        raise ValueError('--action online_infer requires --backbone onlinetas')

    epochs_to_predict = sorted([
        int(f.replace('epoch-', '').replace('.model', ''))
        for f in os.listdir(hasr_model_dir)
        if f.startswith('epoch-') and f.endswith('.model')
    ]) if os.path.isdir(hasr_model_dir) else []

    if not epochs_to_predict:
        raise RuntimeError(f'No refiner checkpoints found in {hasr_model_dir}')

    # Load OnlineTAS backbone at the chosen epoch
    _bb_ckpt = os.path.join(backbone_model_dir, f'epoch-{backbone_predict_epoch}.model')
    backbone.load_state_dict(torch.load(_bb_ckpt, map_location=device, weights_only=False))
    backbone.to(device)
    backbone.eval()

    print(f'Online inference for {len(epochs_to_predict)} refiner epochs: {epochs_to_predict}')
    for ep in epochs_to_predict:
        refiner.load_state_dict(
            torch.load(os.path.join(hasr_model_dir, f'epoch-{ep}.model'),
                       map_location=device, weights_only=False)
        )
        epoch_result_dir = os.path.join(hasr_results_base, f'epoch_{ep}')
        os.makedirs(epoch_result_dir, exist_ok=True)
        print(f'  epoch {ep} …', flush=True)
        run_online_inference(
            hasr_model=refiner, onlinetas_model=backbone, MemoryBank=MemoryBank,
            features_path=features_path, vid_list_file=vid_list_test,
            result_dir=epoch_result_dir, actions_dict=actions_dict,
            device=device, sample_rate=sample_rate, clip_width=args.clip_width,
        )
