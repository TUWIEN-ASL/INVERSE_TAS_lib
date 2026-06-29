"""Shared in-memory evaluation utilities for all TAS models."""

import os
import numpy as np


def _load_gt(gt_path, vid, actions_dict):
    """Load ground-truth labels from .npy or .txt, stripping headers automatically."""
    base = vid.split('.')[0]
    npy_path = os.path.join(gt_path, base + '.npy')
    txt_path = os.path.join(gt_path, vid if '.txt' in vid else base + '.txt')

    if os.path.exists(npy_path):
        ann = np.load(npy_path, allow_pickle=True).astype(int)
        idx_to_action = {v: k for k, v in actions_dict.items()}
        return [idx_to_action[int(x)] for x in ann]

    with open(txt_path) as f:
        content = [ln.strip() for ln in f.read().split('\n') if ln.strip()]
    if content and content[0] not in actions_dict:
        content = content[1:]
    return content


def _get_labels_start_end_time(labels, bg_class=("background",)):
    segs, starts, ends = [], [], []
    last = labels[0]
    if last not in bg_class:
        segs.append(last)
        starts.append(0)
    for i in range(len(labels)):
        if labels[i] != last:
            if labels[i] not in bg_class:
                segs.append(labels[i])
                starts.append(i)
            if last not in bg_class:
                ends.append(i)
            last = labels[i]
    if last not in bg_class:
        ends.append(len(labels))
    return segs, starts, ends


def _levenstein(p, y, norm=True):
    m, n = len(p), len(y)
    if max(m, n) == 0:
        return 100.0
    D = np.zeros([m + 1, n + 1], float)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            D[i, j] = D[i-1, j-1] if y[j-1] == p[i-1] else min(D[i-1, j], D[i, j-1], D[i-1, j-1]) + 1
    return (1 - D[-1, -1] / max(m, n)) * 100 if norm else D[-1, -1]


def _edit_score(recognized, ground_truth, bg_class=("background",)):
    P, _, _ = _get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = _get_labels_start_end_time(ground_truth, bg_class)
    return _levenstein(P, Y)


def _f_score(recognized, ground_truth, overlap, bg_class=("background",)):
    p_lbl, p_s, p_e = _get_labels_start_end_time(recognized, bg_class)
    y_lbl, y_s, y_e = _get_labels_start_end_time(ground_truth, bg_class)
    if not y_lbl:
        return 0.0, float(len(p_lbl)), 0.0
    tp, fp = 0, 0
    hits = np.zeros(len(y_lbl))
    for j in range(len(p_lbl)):
        inter = np.minimum(p_e[j], y_e) - np.maximum(p_s[j], y_s)
        union = np.maximum(p_e[j], y_e) - np.minimum(p_s[j], y_s)
        IoU = (1.0 * inter / union) * [p_lbl[j] == y_lbl[x] for x in range(len(y_lbl))]
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    return float(tp), float(fp), float(len(y_lbl) - sum(hits))


def _dr_f1(gt_labels, pred_labels, threshold=10):
    gt_b = [i for i in range(1, len(gt_labels)) if gt_labels[i] != gt_labels[i-1]]
    pr_b = [i for i in range(1, len(pred_labels)) if pred_labels[i] != pred_labels[i-1]]
    matched = set()
    tp, fp = 0, 0
    for gb in gt_b:
        lo, hi = max(0, gb - threshold), min(len(gt_labels), gb + threshold + 1)
        ms = [i for i, pb in enumerate(pr_b) if lo <= pb < hi]
        if ms:
            best = ms[np.argmin([abs(pr_b[i] - gb) for i in ms])] if len(ms) > 1 else ms[0]
            if best not in matched:
                tp += 1
                matched.add(best)
            fp += len(ms) - 1
    fp += sum(1 for i, pb in enumerate(pr_b)
              if not any(max(0, gb - threshold) <= pb < min(len(gt_labels), gb + threshold + 1)
                         for gb in gt_b))
    fn = sum(1 for gb in gt_b
             if not any(max(0, gb - threshold) <= pb < min(len(gt_labels), gb + threshold + 1)
                        for pb in pr_b))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(100 * f1, 4)


def compute_metrics(predictions, gt_path, vid_list, actions_dict):
    """Compute TAS metrics entirely in memory (no disk writes).

    Args:
        predictions: dict {vid_base_name: [label_str, ...]} at full framerate
        gt_path:     directory containing annotation files
        vid_list:    list of video names from bundle file
        actions_dict: {label_str: class_idx}

    Returns:
        dict with val/acc, val/edit, val/f1_10, val/f1_25, val/f1_50, val/dr
    """
    overlap = [0.1, 0.25, 0.5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    correct, total = 0, 0
    edit_sum, dr_sum, n_vids = 0.0, 0.0, 0

    for vid in vid_list:
        recog = predictions.get(vid.split('.')[0]) or predictions.get(vid)
        if recog is None:
            continue
        try:
            gt = _load_gt(gt_path, vid, actions_dict)
        except Exception:
            continue
        min_len = min(len(recog), len(gt))
        if min_len == 0:
            continue
        recog, gt = recog[:min_len], gt[:min_len]

        correct += sum(g == r for g, r in zip(gt, recog))
        total += min_len
        edit_sum += _edit_score(recog, gt)
        for s in range(3):
            tp1, fp1, fn1 = _f_score(recog, gt, overlap[s])
            tp[s] += tp1; fp[s] += fp1; fn[s] += fn1
        dr_sum += _dr_f1(gt, recog)
        n_vids += 1

    acc  = 100.0 * correct / total if total > 0 else 0.0
    edit = edit_sum / n_vids if n_vids > 0 else 0.0
    dr   = dr_sum / n_vids if n_vids > 0 else 0.0
    f1s  = []
    for s in range(3):
        prec = tp[s] / (tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0.0
        rec  = tp[s] / (tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(np.nan_to_num(f1) * 100)

    return {"val/acc": acc, "val/edit": edit,
            "val/f1_10": f1s[0], "val/f1_25": f1s[1], "val/f1_50": f1s[2],
            "val/dr": dr}
