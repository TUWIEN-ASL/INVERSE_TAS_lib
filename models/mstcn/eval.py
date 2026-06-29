#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse
from configs.paths import project_path
from typing import List, Tuple, Set

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
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
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
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
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def get_action_boundaries(action_labels: List[str]) -> List[int]:
    """
    Extract the indices where action boundaries occur in a sequence of action labels.
    A boundary exists between consecutive labels that are different.
    
    Args:
        action_labels: List of action labels
        
    Returns:
        List of indices where boundaries occur
    """
    boundaries = []
    
    for i in range(1, len(action_labels)):
        if action_labels[i] != action_labels[i-1]:
            boundaries.append(i)
            
    return boundaries

def calculate_detection_rate(ground_truth_labels: List[str], 
                             predicted_labels: List[str], 
                             temporal_threshold: int,
                             verbose: bool = False) -> Tuple[int, int, int]:
    """
    Calculate detection rate metrics based on temporal threshold.
    
    Args:
        ground_truth_labels: List of ground truth action labels
        predicted_labels: List of predicted action labels
        temporal_threshold: Maximum allowed temporal distance for a true positive
        verbose: Whether to print detailed matching information
        
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    # Ensure both lists have the same length
    if len(ground_truth_labels) != len(predicted_labels):
        raise ValueError("Ground truth and prediction lists must have the same length")
    
    # Get boundaries for both ground truth and predictions
    gt_boundaries = get_action_boundaries(ground_truth_labels)
    pred_boundaries = get_action_boundaries(predicted_labels)
    
    if verbose:
        print("Ground truth boundaries:", gt_boundaries)
        print("Predicted boundaries:", pred_boundaries)
    
    true_positives = 0
    false_positives = 0
    
    # Track which predicted boundaries match each ground truth boundary
    # Key: gt_boundary, Value: list of matching pred_boundary indices
    gt_to_pred_matches: Dict[int, List[int]] = {}
    
    # Track which predicted boundaries have been counted as true positives
    matched_as_tp = set()
    
    # For each ground truth boundary, find all predicted boundaries within its window
    for gt_boundary in gt_boundaries:
        # Define the valid window for this ground truth boundary
        window_start = max(0, gt_boundary - temporal_threshold)
        window_end = min(len(ground_truth_labels), gt_boundary + temporal_threshold + 1)
        
        if verbose:
            print(f"\nChecking GT boundary at {gt_boundary} with window [{window_start}, {window_end})")
        
        # Find all predicted boundaries within this window
        matching_preds = []
        for i, pred_boundary in enumerate(pred_boundaries):
            if window_start <= pred_boundary < window_end:
                matching_preds.append(i)
        
        gt_to_pred_matches[gt_boundary] = matching_preds
        
        if verbose:
            print(f"All matching prediction indices: {matching_preds}")
        
        if matching_preds:
            # Case 1: At least one prediction in the window
            
            # Choose the closest prediction as the true positive
            if len(matching_preds) > 1:
                # Calculate distances to the ground truth boundary
                distances = [abs(pred_boundaries[i] - gt_boundary) for i in matching_preds]
                # Get index of the closest prediction
                closest_idx = matching_preds[np.argmin(distances)]
            else:
                closest_idx = matching_preds[0]
            
            # Mark as true positive if not already counted
            if closest_idx not in matched_as_tp:
                true_positives += 1
                matched_as_tp.add(closest_idx)
                
                if verbose:
                    print(f"TP: Matched with prediction at index {closest_idx} (boundary {pred_boundaries[closest_idx]})")
            elif verbose:
                print(f"Already matched prediction at index {closest_idx} - no additional TP")
            
            # Count additional predictions in the window as false positives (CASE 2)
            additional_fps = len(matching_preds) - 1
            false_positives += additional_fps
            
            if verbose and additional_fps > 0:
                print(f"FP: {additional_fps} additional predictions in this window")
        elif verbose:
            print("No matching predictions found - FALSE NEGATIVE")
    
    # Count predictions outside any ground truth window as false positives (CASE 1)
    outside_window_fps = 0
    for i, pred_boundary in enumerate(pred_boundaries):
        # Check if this prediction falls within any ground truth window
        in_any_window = False
        for gt_boundary in gt_boundaries:
            window_start = max(0, gt_boundary - temporal_threshold)
            window_end = min(len(ground_truth_labels), gt_boundary + temporal_threshold + 1)
            if window_start <= pred_boundary < window_end:
                in_any_window = True
                break
        
        if not in_any_window:
            outside_window_fps += 1
            if verbose:
                print(f"FP: Prediction {i} at position {pred_boundary} outside any ground truth window")
    
    false_positives += outside_window_fps
    
    # Calculate false negatives (ground truth boundaries with no matching predictions)
    false_negatives = 0
    for gt_boundary in gt_boundaries:
        if not gt_to_pred_matches.get(gt_boundary, []):
            false_negatives += 1
    
    if verbose:
        print("\nSummary:")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
    
    return true_positives, false_positives, false_negatives

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')
    parser.add_argument('--f_type', default='1')

    args = parser.parse_args()

    ground_truth_path = f"{project_path}/data/"+args.dataset+"/annotations/"
    file_list = f"{project_path}/data/"+args.dataset+"/splits/test.split"+args.split+".bundle"

    top_f150 = 0

    for epoch in range(1, 51):
        recog_path = f"{project_path}/results/mstcn/"+f"{args.dataset}_{args.f_type}"+"/split_"+args.split+f"/epoch_{epoch}"
        # import pdb; pdb.set_trace()
        print(recog_path)
        list_of_videos = read_file(file_list).split('\n')[:-1]

        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct = 0
        total = 0
        edit = 0
        tp_det, fp_det, fn_det = 0, 0, 0

        boundary = 10

        for vid in list_of_videos:
            if ".txt" in vid:
                gt_file = ground_truth_path + vid
            else:
                gt_file = ground_truth_path + vid + ".txt"
            gt_content = read_file(gt_file).split('\n')[0:-1]
            gt_content = gt_content[1:]

            recog_file = recog_path + "/" + vid.split('.')[0]
            recog_content = read_file(recog_file).split('\n')[1].split()

            min_shape = min(len(recog_content), len(gt_content))
            recog_content = recog_content[:min_shape]
            gt_content = gt_content[:min_shape]

            # import pdb; pdb.set_trace()

            # import pdb; pdb.set_trace()

            # for i in range(len(gt_content)):
            #     if "pick" in gt_content[i]:
            #         gt_content[i] = "pick"
            #     if "insert" in gt_content[i]:
            #         gt_content[i] = "insert"
            #     if "remove" in gt_content[i]:
            #         gt_content[i] = "remove"
            #     if "place" in gt_content[i]:
            #         gt_content[i] = "place"
            
            # for i in range(len(recog_content)):
            #     if "pick" in recog_content[i]:
            #         recog_content[i] = "pick"
            #     if "insert" in recog_content[i]:
            #         recog_content[i] = "insert"
            #     if "remove" in recog_content[i]:
            #         recog_content[i] = "remove"
            #     if "place" in recog_content[i]:
            #         recog_content[i] = "place"

            # import pdb; pdb.set_trace()

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

        print("---------------- EPOCH:", epoch , "--------------")
        print("Acc: %.4f" % (100*float(correct)/total))
        print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])
        
            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100
            print('F1@%0.2f: %.4f' % (overlap[s], f1))

        precision = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0
        recall = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0
        f1_bound = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_bound = round(100*f1_bound, 2)
        print("DR", f1_bound)

        if f1 >= top_f150:
            top_f150 = f1
            print("Best result fo far")
        print("---------------------------------------")

if __name__ == '__main__':
    main()