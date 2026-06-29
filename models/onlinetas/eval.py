import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import argparse
from configs.paths import project_path
from typing import List, Tuple, Dict


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels, starts, ends = [], [], []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
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


def get_action_boundaries(action_labels: List[str]) -> List[int]:
    boundaries = []
    for i in range(1, len(action_labels)):
        if action_labels[i] != action_labels[i - 1]:
            boundaries.append(i)
    return boundaries


def calculate_detection_rate(ground_truth_labels, predicted_labels,
                              temporal_threshold, verbose=False):
    if len(ground_truth_labels) != len(predicted_labels):
        raise ValueError("GT and prediction lists must have the same length")

    gt_boundaries = get_action_boundaries(ground_truth_labels)
    pred_boundaries = get_action_boundaries(predicted_labels)

    true_positives = 0
    false_positives = 0
    gt_to_pred_matches: Dict[int, List[int]] = {}
    matched_as_tp = set()

    for gt_boundary in gt_boundaries:
        window_start = max(0, gt_boundary - temporal_threshold)
        window_end = min(len(ground_truth_labels), gt_boundary + temporal_threshold + 1)
        matching_preds = [i for i, pb in enumerate(pred_boundaries)
                          if window_start <= pb < window_end]
        gt_to_pred_matches[gt_boundary] = matching_preds
        if matching_preds:
            if len(matching_preds) > 1:
                distances = [abs(pred_boundaries[i] - gt_boundary) for i in matching_preds]
                closest_idx = matching_preds[np.argmin(distances)]
            else:
                closest_idx = matching_preds[0]
            if closest_idx not in matched_as_tp:
                true_positives += 1
                matched_as_tp.add(closest_idx)
            false_positives += len(matching_preds) - 1

    outside_fps = sum(
        1 for i, pb in enumerate(pred_boundaries)
        if not any(
            max(0, gb - temporal_threshold) <= pb < min(len(ground_truth_labels), gb + temporal_threshold + 1)
            for gb in gt_boundaries
        )
    )
    false_positives += outside_fps
    false_negatives = sum(1 for gb in gt_boundaries if not gt_to_pred_matches.get(gb, []))
    return true_positives, false_positives, false_negatives


def _load_gt(gt_path, vid, actions_dict):
    """Load ground-truth labels, supporting both .npy and .txt formats."""
    base = vid.split('.')[0]
    npy_path = gt_path + base + '.npy'
    txt_path = gt_path + (vid if '.txt' in vid else base + '.txt')

    if os.path.exists(npy_path):
        ann = np.load(npy_path, allow_pickle=True).astype(int)
        idx_to_action = {v: k for k, v in actions_dict.items()}
        return [idx_to_action[int(x)] for x in ann]
    else:
        content = read_file(txt_path).split('\n')
        content = [c for c in content if c]
        if content and content[0] not in actions_dict:
            content = content[1:]
        return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='REASSEMBLEmm')
    parser.add_argument('--split', default='1')
    parser.add_argument('--f_type', default='UM2R_features_bs10_onlyprio_ext_sched')
    args = parser.parse_args()

    ground_truth_path = f"{project_path}/data/{args.dataset}/annotations/"
    file_list = f"{project_path}/data/{args.dataset}/splits/test.split{args.split}.bundle"
    mapping_file = f"{project_path}/data/{args.dataset}/mapping.txt"

    with open(mapping_file, 'r') as f:
        actions = f.read().split('\n')
    actions_dict = {}
    for a in actions:
        parts = a.split()
        if len(parts) >= 2:
            actions_dict[parts[1]] = int(parts[0])

    top_f150 = 0
    best_epoch = None
    best_metrics = None

    for epoch in range(1, 51):
        recog_path = (f"{project_path}/results/onlinetas/"
                      f"{args.dataset}_{args.f_type}/split_{args.split}/epoch_{epoch}")

        if not os.path.exists(recog_path):
            continue

        list_of_videos = read_file(file_list).split('\n')[:-1]

        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
        correct = 0
        total = 0
        edit = 0
        tp_det, fp_det, fn_det = 0, 0, 0
        boundary = 10

        for vid in list_of_videos:
            gt_content = _load_gt(ground_truth_path, vid, actions_dict)

            recog_file = recog_path + "/" + vid.split('.')[0]
            if not os.path.exists(recog_file):
                continue
            recog_content = read_file(recog_file).split('\n')[1].split()

            min_shape = min(len(recog_content), len(gt_content))
            recog_content = recog_content[:min_shape]
            gt_content = gt_content[:min_shape]

            for i in range(len(gt_content)):
                total += 1
                if gt_content[i] == recog_content[i]:
                    correct += 1

            edit += edit_score(recog_content, gt_content)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1

            tp_b, fp_b, fn_b = calculate_detection_rate(gt_content, recog_content, boundary)
            tp_det += tp_b
            fp_det += fp_b
            fn_det += fn_b

        if total == 0:
            continue

        print("---------------- EPOCH:", epoch, "--------------")
        print("Acc: %.4f" % (100 * float(correct) / total))
        print('Edit: %.4f' % ((1.0 * edit) / len(list_of_videos)))
        f1_scores = []
        for s in range(len(overlap)):
            prec = tp[s] / float(tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0
            rec = tp[s] / float(tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0
            f1 = np.nan_to_num(2.0 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0) * 100
            f1_scores.append(f1)
            print('F1@%0.2f: %.4f' % (overlap[s], f1))

        prec = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0
        rec = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0
        f1_bound = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        print("DR", round(100 * f1_bound, 2))

        if f1_scores[2] >= top_f150:
            top_f150 = f1_scores[2]
            best_epoch = epoch
            best_metrics = {
                "acc": 100 * float(correct) / total,
                "edit": (1.0 * edit) / len(list_of_videos),
                "f1_10": f1_scores[0],
                "f1_25": f1_scores[1],
                "f1_50": f1_scores[2],
                "dr": round(100 * f1_bound, 2),
            }
            print("Best result so far")
        print("---------------------------------------")

    if best_epoch is not None:
        m = best_metrics
        print("\n================ BEST EPOCH: %d ================" % best_epoch)
        print("Acc:      %.4f" % m["acc"])
        print("Edit:     %.4f" % m["edit"])
        print("F1@0.10:  %.4f" % m["f1_10"])
        print("F1@0.25:  %.4f" % m["f1_25"])
        print("F1@0.50:  %.4f" % m["f1_50"])
        print("DR:       %.2f" % m["dr"])
        print("=================================================")


if __name__ == '__main__':
    main()
