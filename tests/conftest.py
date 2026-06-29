"""Shared pytest fixtures for the INVERSE_TAS test suite."""

import os
import sys

import numpy as np
import pytest

# Make project root importable when running pytest locally (Docker sets PYTHONPATH=/workspace)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Make helpers.py importable within test files via `from helpers import ...`
sys.path.insert(0, os.path.dirname(__file__))

import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def actions_dict():
    return {"a0": 0, "a1": 1, "a2": 2}


@pytest.fixture
def make_dataset(tmp_path, actions_dict):
    """
    Factory that writes feature .npy files, annotation .txt files, and a
    bundle file for the given list of video base-names.

    Features are stored as (T, D) on disk — all model loaders do np.load().T
    to get (D, T).  Annotations use the header-then-labels format expected by
    BatchGenerator and eval_utils.

    Returns (features_path, gt_path, bundle_file) where features_path and
    gt_path end with '/' for direct concatenation with video names.
    """
    def _make(videos, feat_dim=16, seq_len=50):
        feat_dir = tmp_path / "features"
        gt_dir = tmp_path / "annotations"
        feat_dir.mkdir(exist_ok=True)
        gt_dir.mkdir(exist_ok=True)

        label_list = list(actions_dict.keys())
        for vid in videos:
            feats = np.random.randn(seq_len, feat_dim).astype(np.float32)
            np.save(str(feat_dir / f"{vid}.npy"), feats)

            labels = [label_list[i % len(label_list)] for i in range(seq_len)]
            with open(str(gt_dir / f"{vid}.txt"), "w") as f:
                f.write("### Frame level recognition: ###\n")
                f.write("\n".join(labels) + "\n")

        bundle_file = tmp_path / "test.bundle"
        with open(str(bundle_file), "w") as f:
            f.write("\n".join(videos) + "\n")

        return str(feat_dir) + "/", str(gt_dir) + "/", str(bundle_file)

    return _make
