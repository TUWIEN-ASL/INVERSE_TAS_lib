import argparse
import os
import numpy as np

import sys
_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_dir, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from configs.paths import project_path


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def get_labels_start_end_time(labels, bg_class=('background',)):
    labs, starts, ends = [], [], []
    last = labels[0]
    if last not in bg_class:
        labs.append(last)
        starts.append(0)
    for i in range(len(labels)):
        if labels[i] != last:
            if labels[i] not in bg_class:
                labs.append(labels[i])
                starts.append(i)
            if last not in bg_class:
                ends.append(i)
            last = labels[i]
    if last not in bg_class:
        ends.append(i + 1)
    return labs, starts, ends


def levenstein(p, y, norm=False):
    m, n = len(p), len(y)
    D = np.zeros([m + 1, n + 1], dtype=float)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        return (1 - D[-1, -1] / max(m, n)) * 100
    return D[-1, -1]


def edit_score(recognized, ground_truth, norm=True, bg_class=('background',)):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=('background',)):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp, fp = 0, 0
    hits = np.zeros(len(y_label))
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def evaluate_epoch(dataset, split, f_type, epoch, results_base, verbose=True):
    """Evaluate one epoch of predictions against ground truth.

    Returns a dict with keys: acc, edit, f1_10, f1_25, f1_50.
    Returns None if the result directory does not exist.
    """
    gt_path = f'{project_path}/data/{dataset}/annotations/'
    bundle_prefix = 'hand_test' if 'handcam' in f_type else 'test'
    file_list = f'{project_path}/data/{dataset}/splits/{bundle_prefix}.split{split}.bundle'
    recog_path = os.path.join(results_base, f'epoch_{epoch}')

    if not os.path.isdir(recog_path):
        return None

    videos = [v for v in read_file(file_list).split('\n') if v.strip()]
    overlap = [0.1, 0.25, 0.5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    correct, total, edit = 0, 0, 0.0

    for vid in videos:
        gt_file = gt_path + (vid if '.txt' in vid else vid + '.txt')
        gt = [l for l in read_file(gt_file).split('\n') if l.strip()][1:]

        recog_file = os.path.join(recog_path, vid.split('.')[0])
        if not os.path.isfile(recog_file):
            continue
        recog = read_file(recog_file).split('\n')[1].split()

        min_len = min(len(gt), len(recog))
        gt, recog = gt[:min_len], recog[:min_len]

        correct += sum(g == r for g, r in zip(gt, recog))
        total += min_len
        edit += edit_score(recog, gt)
        for s in range(3):
            tp1, fp1, fn1 = f_score(recog, gt, overlap[s])
            tp[s] += tp1; fp[s] += fp1; fn[s] += fn1

    if total == 0:
        return None

    f1 = []
    for s in range(3):
        prec = tp[s] / (tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0
        rec = tp[s] / (tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0
        f1.append(2.0 * prec * rec / (prec + rec) * 100 if (prec + rec) > 0 else 0)

    metrics = {
        'acc':   100.0 * correct / total,
        'edit':  edit / len(videos),
        'f1_10': f1[0],
        'f1_25': f1[1],
        'f1_50': f1[2],
    }

    if verbose:
        print(f'---------------- EPOCH: {epoch} ----------------')
        print(f'Acc:  {metrics["acc"]:.4f}')
        print(f'Edit: {metrics["edit"]:.4f}')
        for s, thr in enumerate([0.10, 0.25, 0.50]):
            print(f'F1@{thr:.2f}: {f1[s]:.4f}')
        print('--------------------------------------------------')

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ImPerfectPour')
    parser.add_argument('--split', default='1')
    parser.add_argument('--f_type', default='BRP_features')
    parser.add_argument('--backbone', default='mstcn', choices=['mstcn', 'ASFormer', 'onlinetas'])
    parser.add_argument('--refiner',  default='transformer', choices=['transformer', 'gru'])
    parser.add_argument('--online',   action='store_true',
                        help='Evaluate online HASR results (uses _online results dir)')
    parser.add_argument('--num_epochs', type=int, default=50)
    args = parser.parse_args()

    _online_suffix = '_online' if args.online else ''
    results_base = (
        f'{project_path}/results/HASR'
        f'/{args.dataset}_{args.f_type}_{args.backbone}_{args.refiner}{_online_suffix}/split_{args.split}'
    )
    print(f'Evaluating results from: {results_base}')

    for epoch in range(1, args.num_epochs + 1):
        epoch_dir = os.path.join(results_base, f'epoch_{epoch}')
        if os.path.isdir(epoch_dir):
            evaluate_epoch(args.dataset, args.split, args.f_type, epoch, results_base)


if __name__ == '__main__':
    main()
