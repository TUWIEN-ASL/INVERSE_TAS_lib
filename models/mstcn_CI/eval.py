#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse
import os
import glob
import subprocess
import sys
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

def evaluate_checkpoint(epoch_num, ground_truth_path, recog_path, file_list, boundary=10, eval_mode='last'):
    """
    Evaluate a single checkpoint and return metrics.
    
    Args:
        epoch_num: Epoch number for identification
        ground_truth_path: Path to ground truth annotations
        recog_path: Path to recognition results (base path)
        file_list: List of video files to evaluate
        boundary: Temporal boundary threshold
        eval_mode: 'all' or 'last' - determines path structure
        
    Returns:
        Dictionary containing all metrics
    """
    # Determine the actual recognition path based on eval mode
    if eval_mode == 'all':
        actual_recog_path = os.path.join(recog_path, f"epoch_{epoch_num}")
    else:
        actual_recog_path = recog_path
    
    # Check if the recognition directory exists
    if not os.path.exists(actual_recog_path):
        print(f"Warning: Recognition directory not found: {actual_recog_path}")
        return None
    
    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0
    tp_det, fp_det, fn_det = 0, 0, 0

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid + ".txt"
        gt_content = read_file(gt_file).split('\n')[0:-1]
        gt_content = gt_content[1:]

        recog_file = actual_recog_path + "/" + vid.split('.')[0]
        if not os.path.exists(recog_file):
            print(f"Warning: Recognition file not found: {recog_file}")
            continue
            
        recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            if "pick" in gt_content[i]:
                gt_content[i] = "pick"
            if "insert" in gt_content[i]:
                gt_content[i] = "insert"
            if "remove" in gt_content[i]:
                gt_content[i] = "remove"
            if "place" in gt_content[i]:
                gt_content[i] = "place"
        
        for i in range(len(recog_content)):
            if "pick" in recog_content[i]:
                recog_content[i] = "pick"
            if "insert" in recog_content[i]:
                recog_content[i] = "insert"
            if "remove" in recog_content[i]:
                recog_content[i] = "remove"
            if "place" in recog_content[i]:
                recog_content[i] = "place"

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

    acc = 100*float(correct)/total
    edit = (1.0*edit)/len(list_of_videos)
    f1_values = []
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s]) if (tp[s]+fp[s]) > 0 else 0
        recall = tp[s] / float(tp[s]+fn[s]) if (tp[s]+fn[s]) > 0 else 0
        f1 = 2.0 * (precision*recall) / (precision+recall) if (precision+recall) > 0 else 0
        f1 = np.nan_to_num(f1)*100
        f1_values.append(f1)

    precision = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0
    recall = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0
    f1_bound = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_bound = round(100*f1_bound, 2)

    return {
        'epoch': epoch_num,
        'acc': acc,
        'edit': edit,
        'f1_10': f1_values[0],
        'f1_25': f1_values[1],
        'f1_50': f1_values[2],
        'f1_bound': f1_bound
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')
    parser.add_argument('--f_type', default='1')
    parser.add_argument('--eval_mode', default='last', choices=['all', 'last'], 
                       help='Evaluate all checkpoints or just the last one')
    parser.add_argument('--auto_predict', action='store_true',
                       help='Automatically generate predictions if they do not exist')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results to a CSV file')
    parser.add_argument('--save_summary', action='store_true',
                       help='Save a summary file with best metrics and statistics')

    args = parser.parse_args()

    ground_truth_path = f"{project_path}/data/"+args.dataset+"/annotations/"
    model_dir = f"{project_path}/checkpoints/mstcn/"+f"{args.dataset}_{args.f_type}"+"/split_"+args.split
    recog_path = f"{project_path}/results/mstcn/"+f"{args.dataset}_{args.f_type}"+"/split_"+args.split
    file_list = f"{project_path}/data/"+args.dataset+"/splits/test.split"+args.split+".bundle"

    # Get available checkpoints
    checkpoints = get_available_checkpoints(model_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {model_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    
    # Determine which checkpoints to evaluate
    if args.eval_mode == 'last':
        # Evaluate only the last checkpoint
        checkpoints_to_eval = [checkpoints[-1]]
        print(f"Evaluating last checkpoint: epoch {checkpoints[-1][0]}")
    else:
        # Evaluate all checkpoints
        checkpoints_to_eval = checkpoints
        print(f"Evaluating all {len(checkpoints)} checkpoints")
        
        # Check if we need to generate predictions first
        first_epoch_dir = os.path.join(recog_path, f"epoch_{checkpoints[0][0]}")
        if not os.path.exists(first_epoch_dir):
            print(f"\nWarning: Epoch-specific prediction directories not found!")
            
            if args.auto_predict:
                print("Automatically generating predictions for all epochs...")
                try:
                    # Construct the command to generate predictions
                    cmd = [
                        sys.executable, "main.py",
                        "--action", "predict",
                        "--dataset", args.dataset,
                        "--split", args.split,
                        "--f_type", args.f_type,
                        "--predict_mode", "all"
                    ]
                    
                    # Run the prediction command
                    subprocess.run(cmd, check=True)
                    print("Predictions generated successfully!")
                    
                except subprocess.CalledProcessError as e:
                    print(f"Error generating predictions: {e}")
                    return
                except FileNotFoundError:
                    print("Error: main.py not found in current directory")
                    return
            else:
                print(f"It looks like you need to generate predictions for all epochs first.")
                print(f"Please run:")
                print(f"python main.py --action predict --dataset {args.dataset} --split {args.split} --f_type {args.f_type} --predict_mode all")
                print(f"Or add --auto_predict flag to automatically generate them.")
                return

    # Store results for all evaluations
    all_results = []
    
    # Print header
    print("\nEpoch\tACC\tEDIT\tF1@10\tF1@25\tF1@50\tF1_Bound")
    print("-" * 70)
    
    for epoch_num, checkpoint_path in checkpoints_to_eval:
        try:
            # Add debug information
            if args.eval_mode == 'all':
                actual_recog_path = os.path.join(recog_path, f"epoch_{epoch_num}")
                print(f"Evaluating epoch {epoch_num} from: {actual_recog_path}")
            
            results = evaluate_checkpoint(epoch_num, ground_truth_path, recog_path, file_list, eval_mode=args.eval_mode)
            
            if results is None:
                print(f"Skipping epoch {epoch_num} due to missing results")
                continue
                
            all_results.append(results)
            
            # Print results for this epoch
            print(f"{results['epoch']}\t{results['acc']:.4f}\t{results['edit']:.4f}\t"
                  f"{results['f1_10']:.4f}\t{results['f1_25']:.4f}\t{results['f1_50']:.4f}\t"
                  f"{results['f1_bound']:.2f}")
            
        except Exception as e:
            print(f"Error evaluating epoch {epoch_num}: {str(e)}")
            continue
    
    # Print summary statistics if evaluating multiple checkpoints
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        # Find best performing epochs for each metric  
        best_acc = max(all_results, key=lambda x: x['acc'])
        best_edit = max(all_results, key=lambda x: x['edit'])  # Higher is better for normalized edit score
        best_f1_10 = max(all_results, key=lambda x: x['f1_10'])
        best_f1_25 = max(all_results, key=lambda x: x['f1_25']) 
        best_f1_50 = max(all_results, key=lambda x: x['f1_50'])
        best_f1_bound = max(all_results, key=lambda x: x['f1_bound'])
        
        print(f"Best Accuracy: {best_acc['acc']:.4f} (Epoch {best_acc['epoch']})")
        print(f"Best Edit Score: {best_edit['edit']:.4f} (Epoch {best_edit['epoch']})")
        print(f"Best F1@10: {best_f1_10['f1_10']:.4f} (Epoch {best_f1_10['epoch']})")
        print(f"Best F1@25: {best_f1_25['f1_25']:.4f} (Epoch {best_f1_25['epoch']})")
        print(f"Best F1@50: {best_f1_50['f1_50']:.4f} (Epoch {best_f1_50['epoch']})")
        print(f"Best F1_Bound: {best_f1_bound['f1_bound']:.2f} (Epoch {best_f1_bound['epoch']})")
        
        # Calculate average metrics across all epochs
        avg_acc = np.mean([r['acc'] for r in all_results])
        avg_edit = np.mean([r['edit'] for r in all_results])
        avg_f1_10 = np.mean([r['f1_10'] for r in all_results])
        avg_f1_25 = np.mean([r['f1_25'] for r in all_results])
        avg_f1_50 = np.mean([r['f1_50'] for r in all_results])
        avg_f1_bound = np.mean([r['f1_bound'] for r in all_results])
        
        print(f"\nAverage Metrics:")
        print(f"Avg Accuracy: {avg_acc:.4f}")
        print(f"Avg Edit Score: {avg_edit:.4f}")
        print(f"Avg F1@10: {avg_f1_10:.4f}")
        print(f"Avg F1@25: {avg_f1_25:.4f}")
        print(f"Avg F1@50: {avg_f1_50:.4f}")
        print(f"Avg F1_Bound: {avg_f1_bound:.2f}")
    
    # Save results to CSV if requested
    if args.save_results and all_results:
        import csv
        csv_filename = f"evaluation_results_{args.dataset}_{args.f_type}_split{args.split}_{args.eval_mode}.csv"
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'acc', 'edit', 'f1_10', 'f1_25', 'f1_50', 'f1_bound']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
        
        print(f"\nDetailed results saved to {csv_filename}")
    
    # Save summary file if requested
    if args.save_summary and all_results:
        summary_filename = f"evaluation_summary_{args.dataset}_{args.f_type}_split{args.split}_{args.eval_mode}.txt"
        
        with open(summary_filename, 'w') as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Feature Type: {args.f_type}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Evaluation Mode: {args.eval_mode}\n")
            f.write(f"Number of Epochs Evaluated: {len(all_results)}\n")
            f.write(f"Evaluated Epochs: {[r['epoch'] for r in all_results]}\n")
            f.write("\n")
            
            if len(all_results) > 1:
                # Best performing epochs for each metric
                best_acc = max(all_results, key=lambda x: x['acc'])
                best_edit = max(all_results, key=lambda x: x['edit'])  # Higher is better for normalized edit score
                best_f1_10 = max(all_results, key=lambda x: x['f1_10'])
                best_f1_25 = max(all_results, key=lambda x: x['f1_25'])
                best_f1_50 = max(all_results, key=lambda x: x['f1_50'])
                best_f1_bound = max(all_results, key=lambda x: x['f1_bound'])
                
                f.write("BEST PERFORMING EPOCHS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Accuracy: {best_acc['acc']:.4f} (Epoch {best_acc['epoch']})\n")
                f.write(f"Best Edit Score: {best_edit['edit']:.4f} (Epoch {best_edit['epoch']})\n")
                f.write(f"Best F1@10: {best_f1_10['f1_10']:.4f} (Epoch {best_f1_10['epoch']})\n")
                f.write(f"Best F1@25: {best_f1_25['f1_25']:.4f} (Epoch {best_f1_25['epoch']})\n")
                f.write(f"Best F1@50: {best_f1_50['f1_50']:.4f} (Epoch {best_f1_50['epoch']})\n")
                f.write(f"Best F1_Bound: {best_f1_bound['f1_bound']:.2f} (Epoch {best_f1_bound['epoch']})\n")
                f.write("\n")
                
                # Worst performing epochs for each metric
                worst_acc = min(all_results, key=lambda x: x['acc'])
                worst_edit = min(all_results, key=lambda x: x['edit'])  # Lower is worse for normalized edit score
                worst_f1_10 = min(all_results, key=lambda x: x['f1_10'])
                worst_f1_25 = min(all_results, key=lambda x: x['f1_25'])
                worst_f1_50 = min(all_results, key=lambda x: x['f1_50'])
                worst_f1_bound = min(all_results, key=lambda x: x['f1_bound'])
                
                f.write("WORST PERFORMING EPOCHS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Worst Accuracy: {worst_acc['acc']:.4f} (Epoch {worst_acc['epoch']})\n")
                f.write(f"Worst Edit Score: {worst_edit['edit']:.4f} (Epoch {worst_edit['epoch']})\n")
                f.write(f"Worst F1@10: {worst_f1_10['f1_10']:.4f} (Epoch {worst_f1_10['epoch']})\n")
                f.write(f"Worst F1@25: {worst_f1_25['f1_25']:.4f} (Epoch {worst_f1_25['epoch']})\n")
                f.write(f"Worst F1@50: {worst_f1_50['f1_50']:.4f} (Epoch {worst_f1_50['epoch']})\n")
                f.write(f"Worst F1_Bound: {worst_f1_bound['f1_bound']:.2f} (Epoch {worst_f1_bound['epoch']})\n")
                f.write("\n")
                
                # Average metrics across all epochs
                avg_acc = np.mean([r['acc'] for r in all_results])
                avg_edit = np.mean([r['edit'] for r in all_results])
                avg_f1_10 = np.mean([r['f1_10'] for r in all_results])
                avg_f1_25 = np.mean([r['f1_25'] for r in all_results])
                avg_f1_50 = np.mean([r['f1_50'] for r in all_results])
                avg_f1_bound = np.mean([r['f1_bound'] for r in all_results])
                
                # Standard deviations
                std_acc = np.std([r['acc'] for r in all_results])
                std_edit = np.std([r['edit'] for r in all_results])
                std_f1_10 = np.std([r['f1_10'] for r in all_results])
                std_f1_25 = np.std([r['f1_25'] for r in all_results])
                std_f1_50 = np.std([r['f1_50'] for r in all_results])
                std_f1_bound = np.std([r['f1_bound'] for r in all_results])
                
                f.write("AVERAGE METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}\n")
                f.write(f"Average Edit Score: {avg_edit:.4f} ± {std_edit:.4f}\n")
                f.write(f"Average F1@10: {avg_f1_10:.4f} ± {std_f1_10:.4f}\n")
                f.write(f"Average F1@25: {avg_f1_25:.4f} ± {std_f1_25:.4f}\n")
                f.write(f"Average F1@50: {avg_f1_50:.4f} ± {std_f1_50:.4f}\n")
                f.write(f"Average F1_Bound: {avg_f1_bound:.2f} ± {std_f1_bound:.2f}\n")
                f.write("\n")
                
                # Performance trends (if enough epochs)
                if len(all_results) >= 10:
                    # Calculate trends for last 10 epochs vs first 10 epochs
                    first_10 = all_results[:10]
                    last_10 = all_results[-10:]
                    
                    first_avg_acc = np.mean([r['acc'] for r in first_10])
                    last_avg_acc = np.mean([r['acc'] for r in last_10])
                    first_avg_f1_50 = np.mean([r['f1_50'] for r in first_10])
                    last_avg_f1_50 = np.mean([r['f1_50'] for r in last_10])
                    
                    f.write("PERFORMANCE TRENDS\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"First 10 epochs avg accuracy: {first_avg_acc:.4f}\n")
                    f.write(f"Last 10 epochs avg accuracy: {last_avg_acc:.4f}\n")
                    f.write(f"Accuracy improvement: {last_avg_acc - first_avg_acc:+.4f}\n")
                    f.write(f"First 10 epochs avg F1@50: {first_avg_f1_50:.4f}\n")
                    f.write(f"Last 10 epochs avg F1@50: {last_avg_f1_50:.4f}\n")
                    f.write(f"F1@50 improvement: {last_avg_f1_50 - first_avg_f1_50:+.4f}\n")
                    f.write("\n")
                
                # Top 5 epochs overall (based on F1@50)
                top_5_epochs = sorted(all_results, key=lambda x: x['f1_50'], reverse=True)[:5]
                f.write("TOP 5 EPOCHS (by F1@50)\n")
                f.write("-" * 30 + "\n")
                f.write("Rank\tEpoch\tAcc\tEdit\tF1@10\tF1@25\tF1@50\tF1_Bound\n")
                for i, result in enumerate(top_5_epochs, 1):
                    f.write(f"{i}\t{result['epoch']}\t{result['acc']:.4f}\t{result['edit']:.4f}\t"
                           f"{result['f1_10']:.4f}\t{result['f1_25']:.4f}\t{result['f1_50']:.4f}\t"
                           f"{result['f1_bound']:.2f}\n")
                f.write("\n")
                
            else:
                # Single epoch results
                result = all_results[0]
                f.write("SINGLE EPOCH RESULTS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Epoch: {result['epoch']}\n")
                f.write(f"Accuracy: {result['acc']:.4f}\n")
                f.write(f"Edit Score: {result['edit']:.4f}\n")
                f.write(f"F1@10: {result['f1_10']:.4f}\n")
                f.write(f"F1@25: {result['f1_25']:.4f}\n")
                f.write(f"F1@50: {result['f1_50']:.4f}\n")
                f.write(f"F1_Bound: {result['f1_bound']:.2f}\n")
                f.write("\n")
            
            # Add recommendation section
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            if len(all_results) > 1:
                best_overall = max(all_results, key=lambda x: x['f1_50'])
                f.write(f"Recommended model: Epoch {best_overall['epoch']} (highest F1@50: {best_overall['f1_50']:.4f})\n")
                
                # Check for overfitting signs
                if len(all_results) >= 20:
                    early_epochs = all_results[:10]
                    late_epochs = all_results[-10:]
                    early_avg = np.mean([r['f1_50'] for r in early_epochs])
                    late_avg = np.mean([r['f1_50'] for r in late_epochs])
                    
                    if early_avg > late_avg:
                        f.write("⚠️  Potential overfitting detected: Earlier epochs perform better on average\n")
                        best_early = max(early_epochs, key=lambda x: x['f1_50'])
                        f.write(f"Consider using epoch {best_early['epoch']} to avoid overfitting\n")
                    else:
                        f.write("✅ No clear overfitting signs detected\n")
            else:
                f.write("Single epoch evaluation - consider evaluating multiple epochs for comparison\n")
        
        print(f"Summary saved to {summary_filename}")

if __name__ == '__main__':
    main()