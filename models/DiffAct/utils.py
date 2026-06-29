import os
import json
import random
import torch
import math
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter
import matplotlib

import numpy as np
from typing import List, Tuple, Set


# Method 1: Completely random
colors = np.random.rand(69, 4) # RGBA values

# Method 2: Random but more distinct
def generate_distinct_colors(n):
   colors = []
   for i in range(n):
       # Add slight variations to prevent too similar colors
       h = np.random.rand()
       s = 0.5 + np.random.rand() * 0.5  # 0.5-1.0
       v = 0.8 + np.random.rand() * 0.2  # 0.8-1.0
       colors.append(plt.cm.hsv(h))
   return np.array(colors)

colors = [[0.00000000e+00, 9.19488120e-01, 1.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 3.86766999e-01, 1.00000000e+00, 1.00000000e+00],
          [8.08755588e-03, 1.00000000e+00, 4.63248882e-02, 1.00000000e+00],
          [2.31611316e-02, 1.61777368e-02, 1.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 1.00000000e+00, 4.55147040e-01, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 9.27574363e-01, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 3.94853242e-01, 1.00000000e+00],
          [3.08817276e-02, 7.36544854e-04, 1.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 3.17281635e-01, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 7.19118272e-01, 1.00000000e+00],
          [0.00000000e+00, 7.11032029e-01, 1.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 1.00000000e+00, 8.48894953e-01, 1.00000000e+00],
          [1.00000000e+00, 9.26471515e-02, 0.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 2.47796272e-01, 1.00000000e+00, 1.00000000e+00],
          [6.32351667e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 5.33823969e-01, 1.00000000e+00],
          [5.17645576e-01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 5.10662182e-01, 1.00000000e+00],
          [0.00000000e+00, 2.94119847e-01, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 7.41177212e-01, 0.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 4.64338606e-01, 1.00000000e+00],
          [7.26101667e-01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
          [5.44098191e-02, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 5.55882909e-01, 0.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 1.00000000e+00, 2.46692263e-01, 1.00000000e+00],
          [0.00000000e+00, 5.25737726e-01, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 8.56986151e-01, 0.00000000e+00, 1.00000000e+00],
          [7.26101667e-01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 7.11032029e-01, 1.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 1.08825544e-01, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 4.63235757e-01, 0.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 1.00000000e+00, 7.56248386e-01, 1.00000000e+00],
          [1.58081518e-02, 1.00000000e+00, 3.08836963e-02, 1.00000000e+00],
          [5.39704516e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 8.81250788e-01, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 1.40073576e-01, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 1.63235364e-01, 1.00000000e+00],
          [7.48160607e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 2.09558939e-01, 1.00000000e+00],
          [1.22792334e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
          [4.01836637e-01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 6.18384878e-01, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 8.81250788e-01, 1.00000000e+00],
          [9.34557758e-01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 1.00000000e+00, 5.24631966e-01, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 2.55882515e-01, 1.00000000e+00],
          [1.92277698e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
          [3.32351274e-01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 8.80147939e-01, 0.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 2.54779667e-01, 0.00000000e+00, 1.00000000e+00],
          [2.61763061e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 5.79044697e-01, 0.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 6.25019688e-02, 1.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 9.88973483e-01, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 5.79044697e-01, 0.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 1.00000000e+00, 7.09925102e-01, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 9.37500000e-02, 1.00000000e+00],
          [6.78675243e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 1.40073576e-01, 1.00000000e+00],
          [1.00000000e+00, 5.79044697e-01, 0.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 5.09559333e-01, 0.00000000e+00, 1.00000000e+00],
          [5.39704516e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 3.47426818e-01, 0.00000000e+00, 1.00000000e+00],
          [7.02939879e-01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
          [0.00000000e+00, 1.00000000e+00, 6.40440176e-01, 1.00000000e+00],
          [8.40807758e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
          [1.00000000e+00, 0.00000000e+00, 3.02206091e-01, 1.00000000e+00],
          [0.00000000e+00, 1.00000000e+00, 6.40440176e-01, 1.00000000e+00],
          [7.71993971e-03, 4.70601206e-02, 1.00000000e+00, 1.00000000e+00]]
custom_cmap = matplotlib.colors.ListedColormap(colors)

def load_config_file(config_file):

    all_params = json.load(open(config_file))

    if 'result_dir' not in all_params:
        all_params['result_dir'] = 'result'
    
    if 'log_train_results' not in all_params:
        all_params['log_train_results'] = True
    
    if 'soft_label' not in all_params:
        all_params['soft_label'] = None

    if 'postprocess' not in all_params:
        all_params['postprocess'] = {
            'type': None,
            'value': None
        }

    if 'use_instance_norm' not in all_params['encoder_params']:
        all_params['encoder_params']['use_instance_norm'] = False

    if 'detach_decoder' not in all_params['diffusion_params']:
        all_params['diffusion_params']['detach_decoder'] = False

    assert all_params['loss_weights']['encoder_boundary_loss'] == 0

    return all_params


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mode_filter(x, size):
    def modal(P):
        mode = stats.mode(P)
        return mode.mode[0]
    result = generic_filter(x, modal, size)
    return result

############# Modified from ASFormer/MSTCN #################

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
        ends.append(i)
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


def func_eval(label_dir, pred_dir, video_list, downsample, boundary):
    
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    tp_det, fp_det, fn_det = 0, 0, 0

    correct = 0
    total = 0
    edit = 0

    for vid in video_list:
 
        gt_file = os.path.join(label_dir, f'{vid}.txt')
        gt_content = read_file(gt_file).split('\n')[1:-1]
 
        pred_file = os.path.join(pred_dir, f'{vid}.txt')
        pred_content = read_file(pred_file).split('\n')[1].split()

        # for i in range(len(gt_content)):
        #     if "pick" in gt_content[i]:
        #         gt_content[i] = "pick"
        #     if "insert" in gt_content[i]:
        #         gt_content[i] = "insert"
        #     if "remove" in gt_content[i]:
        #         gt_content[i] = "remove"
        #     if "place" in gt_content[i]:
        #         gt_content[i] = "place"
        
        # for i in range(len(pred_content)):
        #     if "pick" in pred_content[i]:
        #         pred_content[i] = "pick"
        #     if "insert" in pred_content[i]:
        #         pred_content[i] = "insert"
        #     if "remove" in pred_content[i]:
        #         pred_content[i] = "remove"
        #     if "place" in pred_content[i]:
        #         pred_content[i] = "place"

        min_size = min(len(gt_content), len(pred_content))
        gt_content = gt_content[:min_size]
        pred_content = pred_content[:min_size]
        assert(len(gt_content) == len(pred_content))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        edit += edit_score(pred_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

        tp_b, fp_b, fn_b = calculate_detection_rate(gt_content, pred_content, boundary)
        tp_det += tp_b
        fp_det += fp_b
        fn_det += fn_b
     
    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(video_list)
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1

    precision = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0
    recall = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0
    f1_bound = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_bound = round(f1_bound*100, 2)
 
    return acc, edit, f1s, f1_bound


############# Visualization #################

def plot_barcode(class_num, gt=None, pred=None, show=True, save_file=None):

    if class_num <= 10:
        color_map = plt.cm.tab10
    elif class_num > 20:
        # color_map = plt.cm.gist_ncar

        # colors = np.vstack((plt.cm.tab20(np.linspace(0, 1, 20)),
        #            plt.cm.tab20b(np.linspace(0, 1, 20)),
        #            plt.cm.tab20c(np.linspace(0, 1, 20)),
        #            plt.cm.Set3(np.linspace(0, 1, 9))))

        # # Create custom colormap
        # color_map = matplotlib.colors.ListedColormap(colors)

        color_map = custom_cmap
    else:
        color_map = plt.cm.tab20

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map, 
                interpolation='nearest', vmin=0, vmax=class_num-1)

    fig = plt.figure(figsize=(18, 4))

    # a horizontal barcode
    if gt is not None:
        ax1 = fig.add_axes([0, 0.45, 1, 0.2], **axprops)
        ax1.set_title('Ground Truth')
        ax1.imshow(gt.reshape((1, -1)), **barprops)

    if pred is not None:
        ax2 = fig.add_axes([0, 0.15, 1, 0.2], **axprops)
        ax2.set_title('Predicted')
        ax2.imshow(pred.reshape((1, -1)), **barprops)

    if save_file is not None:
        fig.savefig(save_file, dpi=400)
    if show:
        plt.show()

    plt.close(fig)