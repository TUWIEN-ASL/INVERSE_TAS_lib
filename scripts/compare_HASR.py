"""
Evaluate all mstcn and HASR checkpoints and compare them.

Prints a per-epoch table for each model, then a best-vs-best summary.

Usage:
    python scripts/compare_HASR.py \
        --dataset REASSEMBLEmm --split 1 --f_type M2R2_features_demo_prio \
        [--backbone mstcn|ASFormer]
"""

import argparse
import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_dir, '..'))
_hasr_dir = os.path.join(_root, 'models', 'HASR')
for p in [_root, _hasr_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.paths import project_path
from eval import evaluate_epoch  # models/HASR/eval.py

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='REASSEMBLEmm')
parser.add_argument('--split',    default='1')
parser.add_argument('--f_type',   default='BRP_features')
parser.add_argument('--backbone', default='mstcn', choices=['mstcn', 'ASFormer', 'onlinetas'])
parser.add_argument('--refiner',  default='gru', choices=['transformer', 'gru'])
parser.add_argument('--online',   action='store_true',
                    help='Compare online HASR variant (uses _online results dir)')
args = parser.parse_args()


# ── Paths & epoch ranges ───────────────────────────────────────────────────────
# Baseline: results from the backbone alone (before HASR refinement)
if args.backbone == 'onlinetas':
    baseline_results = (
        f'{project_path}/results/onlinetas'
        f'/{args.dataset}_{args.f_type}/split_{args.split}'
    )
    baseline_epochs = list(range(1, 51))
elif args.backbone == 'ASFormer':
    baseline_results = (
        f'{project_path}/results/ASFormer'
        f'/{args.dataset}_{args.f_type}/split_{args.split}'
    )
    baseline_epochs = list(range(10, 121, 10))
else:  # mstcn
    baseline_results = (
        f'{project_path}/results/mstcn'
        f'/{args.dataset}_{args.f_type}/split_{args.split}'
    )
    baseline_epochs = list(range(1, 51))

_online_suffix = '_online' if args.online else ''
hasr_results = (
    f'{project_path}/results/HASR'
    f'/{args.dataset}_{args.f_type}_{args.backbone}_{args.refiner}{_online_suffix}/split_{args.split}'
)
hasr_epochs = list(range(1, 61))   # scan up to 50; skips missing epochs

# ── Formatting helpers ────────────────────────────────────────────────────────
COL = 6   # epoch column width
M   = 8   # metric column width
HDR_FMT  = f'{{:>{COL}}} | {{:>{M}}} {{:>{M}}} {{:>{M}}} {{:>{M}}} {{:>{M}}}'
ROW_FMT  = f'{{:>{COL}}} | {{:>{M}.2f}} {{:>{M}.2f}} {{:>{M}.2f}} {{:>{M}.2f}} {{:>{M}.2f}}'
BEST_FMT = f'{{:>{COL}}} | {{:>{M}.2f}} {{:>{M}.2f}} {{:>{M}.2f}} {{:>{M}.2f}} {{:>{M}.2f}}  ← best'
SEP      = '-' * (COL + 3 + (M + 1) * 5)

def _table_header(title):
    print()
    print(f'  {title}')
    print(f'  {SEP}')
    print('  ' + HDR_FMT.format('Epoch', 'Acc', 'Edit', 'F1@.10', 'F1@.25', 'F1@.50'))
    print(f'  {SEP}')

def _eval_all(results_base, epochs, title):
    """Evaluate every available epoch, print a table, return best metrics."""
    _table_header(title)
    all_metrics = {}
    best, best_ep = None, None

    for ep in epochs:
        m = evaluate_epoch(args.dataset, args.split, args.f_type,
                           ep, results_base, verbose=False)
        if m is None:
            continue
        all_metrics[ep] = m
        is_best = best is None or m['f1_50'] > best['f1_50']
        if is_best:
            best, best_ep = m, ep

    if not all_metrics:
        print(f'  {"(no results found)":<{COL + 3 + (M + 1) * 5}}')
        print(f'  {SEP}')
        return None, None

    for ep, m in sorted(all_metrics.items()):
        fmt = BEST_FMT if ep == best_ep else ROW_FMT
        print('  ' + fmt.format(
            ep, m['acc'], m['edit'], m['f1_10'], m['f1_25'], m['f1_50']
        ))

    print(f'  {SEP}')
    return best, best_ep

# ── Run evaluations ────────────────────────────────────────────────────────────
print()
print('=' * 60)
print(f'  Dataset  : {args.dataset}')
print(f'  Split    : {args.split}')
print(f'  Features : {args.f_type}')
print(f'  Backbone : {args.backbone}')
print(f'  Refiner  : {args.refiner}  ({"online" if args.online else "offline"})')
print('=' * 60)

base_metrics, base_ep = _eval_all(
    baseline_results, baseline_epochs,
    f'{args.backbone} baseline — all epochs'
)

hasr_metrics, hasr_ep = _eval_all(
    hasr_results, hasr_epochs,
    f'HASR (refining {args.backbone}) — all epochs'
)

# ── Best-vs-best summary ──────────────────────────────────────────────────────
W = 14
SSEP = '+' + '-' * (W + 2) + '+' + '-' * (W + 2) + '+' + '-' * (W + 2) + '+'

def _cmp_row(label, bv, hv):
    if bv is None or hv is None:
        return f'| {label:<{W}} | {"N/A":>{W}} | {"N/A":>{W}} |'
    delta = hv - bv
    sign  = '+' if delta >= 0 else ''
    return f'| {label:<{W}} | {bv:{W}.2f} | {hv:{W}.2f} ({sign}{delta:.2f}) |'

print()
print('=' * 60)
print(f'  BEST-VS-BEST SUMMARY')
print(f'  {args.backbone} epoch {base_ep}  vs  HASR epoch {hasr_ep}')
print('=' * 60)
print()
print(SSEP)
print(f'| {"Metric":<{W}} | {args.backbone + " (base)":<{W}} | {"HASR (Δ)":<{W}} |')
print(SSEP)

if base_metrics and hasr_metrics:
    print(_cmp_row('Acc (%)',     base_metrics['acc'],   hasr_metrics['acc']))
    print(_cmp_row('Edit',        base_metrics['edit'],  hasr_metrics['edit']))
    print(_cmp_row('F1@0.10 (%)', base_metrics['f1_10'], hasr_metrics['f1_10']))
    print(_cmp_row('F1@0.25 (%)', base_metrics['f1_25'], hasr_metrics['f1_25']))
    print(_cmp_row('F1@0.50 (%)', base_metrics['f1_50'], hasr_metrics['f1_50']))
else:
    print(f'| {"(missing results)":<{W + 36}} |')

print(SSEP)
print()
