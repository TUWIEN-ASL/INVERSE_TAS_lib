"""Unit tests for configs/eval_utils.py — all pure-function logic, no file I/O except
the compute_metrics integration test that uses a tmp_path fixture."""

import os

import numpy as np
import pytest

from configs.eval_utils import (
    _dr_f1,
    _edit_score,
    _f_score,
    _get_labels_start_end_time,
    _levenstein,
    compute_metrics,
)


# ── _levenstein ───────────────────────────────────────────────────────────────

class TestLevenstein:
    def test_identical_sequences_score_100(self):
        assert _levenstein(["A", "B", "C"], ["A", "B", "C"]) == 100.0

    def test_completely_different_sequences(self):
        score = _levenstein(["A", "B", "C"], ["X", "Y", "Z"])
        assert score < 100.0
        assert score >= 0.0

    def test_single_element_match(self):
        assert _levenstein(["A"], ["A"]) == 100.0

    def test_single_element_mismatch(self):
        assert _levenstein(["A"], ["B"]) == 0.0

    def test_empty_sequences_returns_100(self):
        assert _levenstein([], []) == 100.0

    def test_one_insertion(self):
        score = _levenstein(["A", "B"], ["A", "B", "C"])
        # 1 edit in max(2,3)=3 → (1 - 1/3)*100 ≈ 66.7
        assert 60.0 < score < 80.0

    def test_unnormalized_returns_raw_distance(self):
        d = _levenstein(["A", "B"], ["A", "C"], norm=False)
        assert d == 1.0


# ── _get_labels_start_end_time ────────────────────────────────────────────────

class TestGetLabelsStartEndTime:
    def test_single_segment(self):
        segs, starts, ends = _get_labels_start_end_time(["A", "A", "A"])
        assert segs == ["A"]
        assert starts == [0]
        assert ends == [3]

    def test_two_segments(self):
        segs, starts, ends = _get_labels_start_end_time(["A", "A", "B", "B"])
        assert segs == ["A", "B"]
        assert starts == [0, 2]
        assert ends == [2, 4]

    def test_three_segments(self):
        segs, starts, ends = _get_labels_start_end_time(["A", "B", "A"])
        assert len(segs) == 3
        assert segs == ["A", "B", "A"]

    def test_background_excluded(self):
        segs, starts, ends = _get_labels_start_end_time(
            ["background", "A", "A", "background"],
            bg_class=("background",),
        )
        assert "background" not in segs
        assert segs == ["A"]


# ── _edit_score ───────────────────────────────────────────────────────────────

class TestEditScore:
    def test_perfect_prediction_score_100(self):
        labels = ["A"] * 10 + ["B"] * 10
        assert _edit_score(labels, labels) == 100.0

    def test_reversed_segments_lower_score(self):
        gt = ["A"] * 10 + ["B"] * 10
        pred = ["B"] * 10 + ["A"] * 10
        score = _edit_score(pred, gt)
        assert score < 100.0


# ── _f_score ──────────────────────────────────────────────────────────────────

class TestFScore:
    def test_perfect_overlap_counts_as_tp(self):
        # One segment, identical prediction
        gt = ["A"] * 20 + ["B"] * 20
        tp, fp, fn = _f_score(gt, gt, overlap=0.5)
        assert tp == 2
        assert fn == 0

    def test_no_overlap_counts_as_fp(self):
        gt = ["A"] * 20
        pred = ["B"] * 20  # different label — no IoU match
        tp, fp, fn = _f_score(pred, gt, overlap=0.5)
        assert tp == 0

    def test_empty_gt_returns_zeros(self):
        pred = ["A"] * 10
        # _get_labels_start_end_time on all-background returns empty y_lbl
        tp, fp, fn = _f_score(pred, ["background"] * 10, overlap=0.1)
        assert tp == 0.0
        assert fn == 0.0


# ── _dr_f1 ────────────────────────────────────────────────────────────────────

class TestDrF1:
    def test_perfect_boundary_detection(self):
        gt = ["A"] * 10 + ["B"] * 10
        pred = ["A"] * 10 + ["B"] * 10
        f1 = _dr_f1(gt, pred)
        assert f1 == 100.0

    def test_no_boundaries_in_prediction(self):
        gt = ["A"] * 10 + ["B"] * 10
        pred = ["A"] * 20  # no boundary predicted
        f1 = _dr_f1(gt, pred)
        assert f1 == 0.0

    def test_boundary_within_threshold(self):
        gt = ["A"] * 10 + ["B"] * 10
        # shift boundary by 5 frames (within default threshold=10)
        pred = ["A"] * 15 + ["B"] * 5
        f1 = _dr_f1(gt, pred, threshold=10)
        assert f1 > 0.0

    def test_boundary_outside_threshold(self):
        gt = ["A"] * 10 + ["B"] * 10
        # shift boundary by 15 frames (outside default threshold=10)
        pred = ["A"] * 25 + ["B"] * 5
        f1 = _dr_f1(gt, pred, threshold=10)
        assert f1 == 0.0


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_expected_keys(self, tmp_path, actions_dict):
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        labels = ["a0"] * 20 + ["a1"] * 20 + ["a2"] * 10
        (gt_dir / "vid1.txt").write_text(
            "### Frame level recognition: ###\n" + "\n".join(labels) + "\n"
        )

        predictions = {"vid1": labels}
        vid_list = ["vid1"]
        metrics = compute_metrics(predictions, str(gt_dir) + "/", vid_list, actions_dict)

        assert set(metrics.keys()) == {
            "val/acc", "val/edit", "val/f1_10", "val/f1_25", "val/f1_50", "val/dr"
        }

    def test_perfect_prediction_gives_max_acc(self, tmp_path, actions_dict):
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        labels = ["a0"] * 20 + ["a1"] * 20 + ["a2"] * 10
        (gt_dir / "vid1.txt").write_text(
            "### Frame level recognition: ###\n" + "\n".join(labels) + "\n"
        )

        predictions = {"vid1": labels}
        vid_list = ["vid1"]
        metrics = compute_metrics(predictions, str(gt_dir) + "/", vid_list, actions_dict)

        assert metrics["val/acc"] == pytest.approx(100.0)
        assert metrics["val/edit"] == pytest.approx(100.0)

    def test_wrong_prediction_reduces_acc(self, tmp_path, actions_dict):
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        labels = ["a0"] * 50
        (gt_dir / "vid1.txt").write_text(
            "### Frame level recognition: ###\n" + "\n".join(labels) + "\n"
        )

        wrong_pred = ["a1"] * 50  # all wrong
        metrics = compute_metrics(
            {"vid1": wrong_pred}, str(gt_dir) + "/", ["vid1"], actions_dict
        )
        assert metrics["val/acc"] == pytest.approx(0.0)

    def test_missing_prediction_is_skipped_gracefully(self, tmp_path, actions_dict):
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        labels = ["a0"] * 20
        (gt_dir / "vid1.txt").write_text(
            "### Frame level recognition: ###\n" + "\n".join(labels) + "\n"
        )

        # predictions dict is empty — vid1 has no entry
        metrics = compute_metrics({}, str(gt_dir) + "/", ["vid1"], actions_dict)
        assert metrics["val/acc"] == pytest.approx(0.0)

    def test_npy_ground_truth_loaded_correctly(self, tmp_path, actions_dict):
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        label_indices = np.array([0] * 20 + [1] * 20 + [2] * 10, dtype=int)
        np.save(str(gt_dir / "vid1.npy"), label_indices)

        idx_to_label = {v: k for k, v in actions_dict.items()}
        gt_labels = [idx_to_label[i] for i in label_indices]
        metrics = compute_metrics(
            {"vid1": gt_labels}, str(gt_dir) + "/", ["vid1"], actions_dict
        )
        assert metrics["val/acc"] == pytest.approx(100.0)
