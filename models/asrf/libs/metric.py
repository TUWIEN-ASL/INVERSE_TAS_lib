# https://github.com/cthincsl/TemporalConvthutionalNetworks/blob/master/code/metrics.py
# Score metric for action segmentation was originally written by cthincs1

import copy
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from typing import List, Tuple, Set

def get_segments(
    frame_wise_label: np.ndarray,
    id2class_map: Dict[int, str],
    bg_class: str = "background",
) -> Tuple[List[int], List[int], List[int]]:
    """
    Args:
        frame-wise label: frame-wise prediction or ground truth. 1D numpy array
    Return:
        segment-label array: list (excluding background class)
        start index list
        end index list
    """

    labels = []
    starts = []
    ends = []

    frame_wise_label = [
        id2class_map[frame_wise_label[i]] for i in range(len(frame_wise_label))
    ]

    # get class, start index and end index of segments
    # background class is excluded
    last_label = frame_wise_label[0]
    if frame_wise_label[0] != bg_class:
        labels.append(frame_wise_label[0])
        starts.append(0)

    for i in range(len(frame_wise_label)):
        # if action labels change
        if frame_wise_label[i] != last_label:
            # if label change from one class to another class
            # it's an action starting point
            if frame_wise_label[i] != bg_class:
                labels.append(frame_wise_label[i])
                starts.append(i)

            # if label change from background to a class
            # it's not an action end point.
            if last_label != bg_class:
                ends.append(i)

            # update last label
            last_label = frame_wise_label[i]

    if last_label != bg_class:
        ends.append(i)

    return labels, starts, ends


def levenshtein(pred: List[int], gt: List[int], norm: bool = True) -> float:
    """
    Levenshtein distance(Edit Distance)
    Args:
        pred: segments list
        gt: segments list
    Return:
        if norm == True:
            (1 - average_edit_distance) * 100
        else:
            edit distance
    """

    n, m = len(pred), len(gt)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # insertion
                dp[i][j - 1] + 1,  # deletion
                dp[i - 1][j - 1] + cost,
            )  # replacement

    if norm:
        score = (1 - dp[n][m] / max(n, m)) * 100
    else:
        score = dp[n][m]

    return score


def get_n_samples(
    p_label: List[int],
    p_start: List[int],
    p_end: List[int],
    g_label: List[int],
    g_start: List[int],
    g_end: List[int],
    iou_threshold: float,
    bg_class: List[str] = ["background"],
) -> Tuple[int, int, int]:
    """
    Args:
        p_label, p_start, p_end: return values of get_segments(pred)
        g_label, g_start, g_end: return values of get_segments(gt)
        threshold: threshold (0.1, 0.25, 0.5)
        bg_class: background class
    Return:
        tp: true positive
        fp: false positve
        fn: false negative
    """

    tp = 0
    fp = 0
    hits = np.zeros(len(g_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], g_end) - np.maximum(p_start[j], g_start)
        union = np.maximum(p_end[j], g_end) - np.minimum(p_start[j], g_start)
        IoU = (1.0 * intersection / union) * (
            [p_label[j] == g_label[x] for x in range(len(g_label))]
        )
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= iou_threshold and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = len(g_label) - sum(hits)

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

class ScoreMeter(object):
    def __init__(
        self,
        id2class_map: Dict[int, str],
        iou_thresholds: Tuple[float] = (0.1, 0.25, 0.5),
        ignore_index: int = 255,
    ) -> None:

        self.iou_thresholds = iou_thresholds  # threshold for f score
        self.ignore_index = ignore_index
        self.id2class_map = id2class_map
        self.edit_score = 0
        self.tp = [0 for _ in range(len(iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.n_classes = len(self.id2class_map)
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.tp_det = 0
        self.fp_det = 0
        self.fn_det = 0

    def _fast_hist(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        mask = (gt >= 0) & (gt < self.n_classes)
        hist = np.bincount(
            self.n_classes * gt[mask].astype(int) + pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(
        self,
        outputs: np.ndarray,
        gts: np.ndarray,
        boundaries: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            outputs: np.array. shape(N, C, T)
                the model output for boundary prediciton
            gt: np.array. shape(N, T)
                Ground Truth for boundary
        """
        if len(outputs.shape) == 3:
            preds = outputs.argmax(axis=1)
        elif len(outputs.shape) == 2:
            preds = copy.copy(outputs)

        for pred, gt in zip(preds, gts):
            pred = pred[gt != self.ignore_index]
            gt = gt[gt != self.ignore_index]

            # for i in range(len(gt)):
            #     if gt[i] in list(range(1, 18)):
            #         gt[i] = 1
            #     if gt[i] in list(range(18, 35)):
            #         gt[i] = 2
            #     if gt[i] in list(range(35, 52)):
            #         gt[i] = 3
            #     if gt[i] in list(range(52, 69)):
            #         gt[i] = 4
            
            # for i in range(len(pred)):
            #     if pred[i] in list(range(1, 18)):
            #         pred[i] = 1
            #     if pred[i] in list(range(18, 35)):
            #         pred[i] = 2
            #     if pred[i] in list(range(35, 52)):
            #         pred[i] = 3
            #     if pred[i] in list(range(52, 69)):
            #         pred[i] = 4

            for lt, lp in zip(pred, gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

            self.n_videos += 1
            # count the correct frame
            self.n_frames += len(pred)
            for i in range(len(pred)):
                if pred[i] == gt[i]:
                    self.n_correct += 1

            # calculate the edit distance
            p_label, p_start, p_end = get_segments(pred, self.id2class_map)
            g_label, g_start, g_end = get_segments(gt, self.id2class_map)

            self.edit_score += levenshtein(p_label, g_label, norm=True)

            for i, th in enumerate(self.iou_thresholds):
                tp, fp, fn = get_n_samples(
                    p_label, p_start, p_end, g_label, g_start, g_end, th
                )
                self.tp[i] += tp
                self.fp[i] += fp
                self.fn[i] += fn

            tp_b, fp_b, fn_b = calculate_detection_rate(gt, pred, 10)
            self.tp_det += tp_b
            self.fp_det += fp_b
            self.fn_det += fn_b

    def get_scores(self) -> Tuple[float, float, float]:
        """
        Return:
            Accuracy
            Normlized Edit Distance
            F1 Score of Each Threshold
        """

        # accuracy
        acc = 100 * float(self.n_correct) / self.n_frames

        # edit distance
        edit_score = float(self.edit_score) / self.n_videos

        # F1 Score
        f1s = []
        for i in range(len(self.iou_thresholds)):
            precision = self.tp[i] / float(self.tp[i] + self.fp[i])
            recall = self.tp[i] / float(self.tp[i] + self.fn[i])

            f1 = 2.0 * (precision * recall) / (precision + recall + 1e-7)
            f1 = np.nan_to_num(f1) * 100

            f1s.append(f1)

        precision = self.tp_det / (self.tp_det + self.fp_det) if (self.tp_det + self.fp_det) > 0 else 0
        recall = self.tp_det / (self.tp_det + self.fn_det) if (self.tp_det + self.fn_det) > 0 else 0
        f1_bound = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_bound = round(100*f1_bound, 2)

        # Accuracy, Edit Distance, F1 Score
        return acc, edit_score, f1s, f1_bound

    def return_confusion_matrix(self) -> np.ndarray:
        return self.confusion_matrix

    def save_scores(self, save_path: str) -> None:
        acc, edit_score, segment_f1s, f1_bound = self.get_scores()

        # save log
        columns = ["cls_acc", "edit"]
        data_dict = {
            "cls_acc": [acc],
            "edit": [edit_score],
            "f1_bound": [f1_bound],
        }

        for i in range(len(self.iou_thresholds)):
            key = "segment f1s@{}".format(self.iou_thresholds[i])
            columns.append(key)
            data_dict[key] = [segment_f1s[i]]

        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)

    def save_confusion_matrix(self, save_path: str) -> None:
        with open(save_path, "w") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerows(self.confusion_matrix)

    def reset(self) -> None:
        self.edit_score = 0
        self.tp = [0 for _ in range(len(self.iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(self.iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(self.iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def argrelmax(prob: np.ndarray, threshold: float = 0.7) -> List[int]:
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold

    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0

    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )

    peak_idx = np.where(peak)[0].tolist()

    return peak_idx


class BoundaryScoreMeter(object):
    def __init__(self, tolerance=5, boundary_threshold=0.7):
        # max distance of the frame which can be regarded as correct
        self.tolerance = tolerance

        # threshold of the boundary value which can be regarded as action boundary
        self.boundary_threshold = boundary_threshold
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0

    def update(self, preds, gts, masks):
        """
        Args:
            preds: np.array. the model output(N, T)
            gts: np.array. boudnary ground truth array (N, T)
            masks: np.array. np.bool. valid length for each video (N, T)
        Return:
            Accuracy
            Boundary F1 Score
        """

        for pred, gt, mask in zip(preds, gts, masks):
            # ignore invalid frames
            pred = pred[mask]
            gt = gt[mask]

            pred_idx = argrelmax(pred, threshold=self.boundary_threshold)
            gt_idx = argrelmax(gt, threshold=self.boundary_threshold)

            n_frames = pred.shape[0]
            tp = 0.0
            fp = 0.0
            fn = 0.0

            hits = np.zeros(len(gt_idx))

            # calculate true positive, false negative, false postive, true negative
            for i in range(len(pred_idx)):
                dist = np.abs(np.array(gt_idx) - pred_idx[i])
                min_dist = np.min(dist)
                idx = np.argmin(dist)

                if min_dist <= self.tolerance and hits[idx] == 0:
                    tp += 1
                    hits[idx] = 1
                else:
                    fp += 1

            fn = len(gt_idx) - sum(hits)
            tn = n_frames - tp - fp - fn

            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.n_frames += n_frames
            self.n_correct += tp + tn

    def get_scores(self):
        """
        Return:
            Accuracy
            Boundary F1 Score
        """

        # accuracy
        acc = 100 * self.n_correct / self.n_frames

        # Boudnary F1 Score
        precision = self.tp / float(self.tp + self.fp)
        recall = self.tp / float(self.tp + self.fn)

        f1s = 2.0 * (precision * recall) / (precision + recall + 1e-7)
        f1s = np.nan_to_num(f1s) * 100

        # Accuracy, Edit Distance, F1 Score
        return acc, precision * 100, recall * 100, f1s

    def save_scores(self, save_path: str) -> None:
        acc, precision, recall, f1s = self.get_scores()

        # save log
        columns = ["bound_acc", "precision", "recall", "bound_f1s"]
        data_dict = {
            "bound_acc": [acc],
            "precision": [precision],
            "recall": [recall],
            "bound_f1s": [f1s],
        }

        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)

    def reset(self):
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
