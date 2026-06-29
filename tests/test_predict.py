"""Unit tests for scripts/predict.py — Predictor.save_pred() output format and
the end-to-end predict() pipeline with mocked feature extractor and segmentation
model.

The I3D and BridgePrompt extractors require GPU and large model checkpoints and
are mocked at the sys.modules level before the script is loaded so the tests
run in any environment (including the Docker training container).
"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_BASE = os.path.join(os.path.dirname(__file__), "..")

# ── Mock heavy dependencies before loading predict.py ────────────────────────
# Only mock modules that genuinely can't be imported in the test environment:
#   - models.BridgePrompt.extractor: this file does not exist on disk
#   - models.video_features.models.i3d.*: I3D needs RAFT + OpenCV video I/O
#   - models.video_features.models.raft.*: RAFT has native C extensions
#
# Deliberately NOT mocking parent packages (models.video_features,
# models.video_features.utils, etc.) so that other test files can import the
# real utility modules from those packages.

_HEAVY_MODULES = [
    "models.BridgePrompt",
    "models.BridgePrompt.extractor",
    "models.video_features.models.raft.raft_src",
    "models.video_features.models.raft.raft_src.raft",
    "models.video_features.models.raft.raft_src.corr_fast",
    "models.video_features.models.raft.extract_raft",
    "models.video_features.models.i3d.i3d_src",
    "models.video_features.models.i3d.i3d_src.i3d_net",
    "models.video_features.models.i3d.extract_i3d",
]
for _m in _HEAVY_MODULES:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()


def _load(unique_name, relpath):
    path = os.path.join(_BASE, relpath)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


_predict_mod = _load("scripts_predict", "scripts/predict.py")
Predictor = _predict_mod.Predictor

# ── Helpers ───────────────────────────────────────────────────────────────────

ACTIONS = {"pour": 0, "stir": 1, "cut": 2}


def _make_predictor(tmp_path):
    """Build a minimal Predictor, bypassing __init__ entirely.

    We set only the attributes that the tested methods actually use.
    """
    p = Predictor.__new__(Predictor)
    p.actions_dict = ACTIONS
    p.device = "cpu"
    return p


def _mapping_file(tmp_path):
    """Write a mapping.txt file in the format expected by Predictor.__init__."""
    f = tmp_path / "mapping.txt"
    f.write_text("0 pour\n1 stir\n2 cut\n")
    return str(f)


# ── save_pred ─────────────────────────────────────────────────────────────────

class TestSavePred:
    def test_creates_output_file(self, tmp_path):
        pred = _make_predictor(tmp_path)
        preds = torch.tensor([0, 1, 2])
        out = tmp_path / "result.txt"
        pred.save_pred(preds, str(out))
        assert out.exists()

    def test_output_is_space_separated_labels(self, tmp_path):
        pred = _make_predictor(tmp_path)
        preds = torch.tensor([0, 1, 2])
        out = tmp_path / "result.txt"
        pred.save_pred(preds, str(out))
        content = out.read_text()
        assert content == "pour stir cut"

    def test_single_label(self, tmp_path):
        pred = _make_predictor(tmp_path)
        preds = torch.tensor([1])
        out = tmp_path / "result.txt"
        pred.save_pred(preds, str(out))
        assert out.read_text() == "stir"

    def test_repeated_labels(self, tmp_path):
        pred = _make_predictor(tmp_path)
        preds = torch.tensor([0, 0, 2, 2])
        out = tmp_path / "result.txt"
        pred.save_pred(preds, str(out))
        assert out.read_text() == "pour pour cut cut"

    def test_overwrite_existing_file(self, tmp_path):
        pred = _make_predictor(tmp_path)
        out = tmp_path / "result.txt"
        out.write_text("old content")
        pred.save_pred(torch.tensor([1]), str(out))
        assert out.read_text() == "stir"


# ── predict pipeline ──────────────────────────────────────────────────────────

class TestPredict:
    @pytest.fixture
    def predictor_with_mocks(self, tmp_path):
        p = _make_predictor(tmp_path)
        # Mock the feature extractor: returns a (D, T) tensor
        features = torch.randn(16, 30)
        p.f_extractor = MagicMock(return_value=features)
        # Mock the segmentation model: returns integer class predictions
        p.seg_model = MagicMock()
        p.seg_model.predict.return_value = torch.tensor([0, 1, 2, 0, 1])
        p.stack_size = 21
        p.step_size = 750
        return p

    def test_predict_calls_feature_extractor_with_path(
        self, predictor_with_mocks, tmp_path
    ):
        video_path = str(tmp_path / "fake.mp4")
        predictor_with_mocks.predict(video_path)
        predictor_with_mocks.f_extractor.assert_called_once_with(video_path)

    def test_predict_calls_seg_model_predict(
        self, predictor_with_mocks, tmp_path
    ):
        predictor_with_mocks.predict(str(tmp_path / "fake.mp4"))
        predictor_with_mocks.seg_model.predict.assert_called_once()

    def test_predict_returns_tensor(self, predictor_with_mocks, tmp_path):
        result = predictor_with_mocks.predict(str(tmp_path / "fake.mp4"))
        assert isinstance(result, torch.Tensor)

    def test_predict_save_true_writes_txt_file(
        self, predictor_with_mocks, tmp_path
    ):
        video_path = str(tmp_path / "fake.mp4")
        predictor_with_mocks.predict(video_path, save=True)
        expected = tmp_path / "fake.txt"
        assert expected.exists()

    def test_predict_save_false_does_not_write_file(
        self, predictor_with_mocks, tmp_path
    ):
        video_path = str(tmp_path / "fake.mp4")
        predictor_with_mocks.predict(video_path, save=False)
        assert not (tmp_path / "fake.txt").exists()
