"""Unit tests for models/HASR/ — SparseSampleEmbedder, BaseHASR helpers,
GRURefiner, TransformerHASR forward passes, and the train_refiner loop."""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from helpers import MockBatchGenerator

_BASE = os.path.join(os.path.dirname(__file__), "..")


def _load(unique_name, relpath):
    path = os.path.join(_BASE, relpath)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


_hasr = _load("hasr_model", "models/HASR/model.py")
_hasr_train = _load("hasr_train", "models/HASR/train.py")

BaseHASR = _hasr.BaseHASR
GRURefiner = _hasr.GRURefiner
SparseSampleEmbedder = _hasr.SparseSampleEmbedder
TransformerHASR = _hasr.TransformerHASR
train_refiner = _hasr_train.train_refiner

# ── Hyperparameter defaults for fast tests ────────────────────────────────────

NUM_ACTIONS = 4
INPUT_DIM = 32
FEAT_DIM = 32           # feature dim for segment embeddings
T = 30                  # total video frames
NUM_HL_FRAMES = 4       # power of 2 (log2(4)=2 residual blocks)
NUM_HL_SAMPLES = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_segments(n_frames, num_segments=3):
    """(n_frames,) int tensor with `num_segments` consecutive equal-label runs."""
    seg_len = n_frames // num_segments
    parts = []
    for i in range(num_segments):
        length = seg_len if i < num_segments - 1 else n_frames - seg_len * (num_segments - 1)
        parts.append(torch.full((length,), i % NUM_ACTIONS, dtype=torch.long))
    return torch.cat(parts)


class _DummyBackbone(torch.nn.Module):
    """Minimal backbone compatible with HASR's train.py: returns [logits]."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = torch.nn.Conv1d(input_dim, num_classes, 1)

    def forward(self, x, mask):
        return [self.fc(x)]


# ── SparseSampleEmbedder ──────────────────────────────────────────────────────

class TestSparseSampleEmbedder:
    @pytest.fixture
    def embedder(self):
        return SparseSampleEmbedder(
            in_dim=FEAT_DIM * 2, feat_dim=FEAT_DIM,
            num_frames=NUM_HL_FRAMES, num_samples=NUM_HL_SAMPLES,
        )

    def test_output_shape_standard(self, embedder):
        inp = torch.randn(1, 10, FEAT_DIM * 2)
        assert embedder(inp).shape == (1, FEAT_DIM)

    def test_output_shape_batch_gt1(self, embedder):
        inp = torch.randn(2, 10, FEAT_DIM * 2)
        assert embedder(inp).shape == (2, FEAT_DIM)

    def test_fewer_segments_than_num_frames(self, embedder):
        """L < num_frames → sampling with replacement must not crash."""
        inp = torch.randn(1, 2, FEAT_DIM * 2)
        assert embedder(inp).shape == (1, FEAT_DIM)


# ── BaseHASR (tested via GRURefiner) ─────────────────────────────────────────

class TestBaseHASR:
    @pytest.fixture
    def refiner(self):
        m = GRURefiner(
            num_actions=NUM_ACTIONS, input_dim=INPUT_DIM, feat_dim=FEAT_DIM,
            num_gru_layers=1, num_highlevel_frames=NUM_HL_FRAMES,
            num_highlevel_samples=NUM_HL_SAMPLES, device="cpu",
        )
        m.train()
        return m

    def test_get_segment_info_segment_count(self, refiner):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        seg_idx, seg_feat, _, _ = refiner.get_segment_info(action_idx, batch_input)
        assert len(seg_idx) == 4       # 3 segments → 4 boundary indices
        assert seg_feat.shape[1] == 3

    def test_get_segment_info_feat_dim(self, refiner):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        _, seg_feat, _, _ = refiner.get_segment_info(action_idx, batch_input)
        assert seg_feat.shape == (1, 3, FEAT_DIM)

    def test_rollout_restores_frame_count(self, refiner):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        seg_idx, _, _, _ = refiner.get_segment_info(action_idx, batch_input)
        S = len(seg_idx) - 1
        fake_pred = torch.randn(1, S, NUM_ACTIONS)
        rollout = refiner.rollout(seg_idx, fake_pred)
        assert rollout.shape == (1, NUM_ACTIONS, T)

    def test_get_segment_info_gt_pred_labels_same_shape(self, refiner):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        batch_target = torch.randint(0, NUM_ACTIONS, (1, T))
        _, _, pred_labels, gt_labels = refiner.get_segment_info(
            action_idx, batch_input, batch_target
        )
        assert gt_labels.shape == pred_labels.shape


# ── GRURefiner ────────────────────────────────────────────────────────────────

class TestGRURefiner:
    @pytest.fixture
    def model(self):
        m = GRURefiner(
            num_actions=NUM_ACTIONS, input_dim=INPUT_DIM, feat_dim=FEAT_DIM,
            num_gru_layers=1, num_highlevel_frames=NUM_HL_FRAMES,
            num_highlevel_samples=NUM_HL_SAMPLES, device="cpu",
        )
        m.train()
        return m

    def test_rollout_shape_matches_input_frames(self, model):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        batch_target = torch.randint(0, NUM_ACTIONS, (1, T))
        _, refine_rollout, _ = model(action_idx, batch_input, batch_target)
        assert refine_rollout.shape == (1, NUM_ACTIONS, T)

    def test_pred_has_num_actions_classes(self, model):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        batch_target = torch.randint(0, NUM_ACTIONS, (1, T))
        refine_pred, _, _ = model(action_idx, batch_input, batch_target)
        assert refine_pred.shape[2] == NUM_ACTIONS

    def test_gt_labels_match_pred_segment_count(self, model):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        batch_target = torch.randint(0, NUM_ACTIONS, (1, T))
        refine_pred, _, gt_labels = model(action_idx, batch_input, batch_target)
        assert gt_labels.shape == (1, refine_pred.shape[1])

    def test_gradients_flow(self, model):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        batch_target = torch.randint(0, NUM_ACTIONS, (1, T))
        refine_pred, _, gt_labels = model(action_idx, batch_input, batch_target)
        loss = torch.nn.CrossEntropyLoss()(refine_pred[0], gt_labels.view(-1))
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


# ── TransformerHASR ───────────────────────────────────────────────────────────

class TestTransformerHASR:
    @pytest.fixture
    def model(self):
        m = TransformerHASR(
            num_actions=NUM_ACTIONS, input_dim=INPUT_DIM, feat_dim=FEAT_DIM,
            d_model=32, nhead=4, num_transformer_layers=2, dim_feedforward=64,
            dropout=0.0, num_highlevel_frames=NUM_HL_FRAMES,
            num_highlevel_samples=NUM_HL_SAMPLES, device="cpu",
        )
        m.train()
        return m

    def test_rollout_shape(self, model):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        batch_target = torch.randint(0, NUM_ACTIONS, (1, T))
        _, refine_rollout, _ = model(action_idx, batch_input, batch_target)
        assert refine_rollout.shape == (1, NUM_ACTIONS, T)

    def test_pred_has_num_actions_classes(self, model):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        batch_target = torch.randint(0, NUM_ACTIONS, (1, T))
        refine_pred, _, _ = model(action_idx, batch_input, batch_target)
        assert refine_pred.shape[2] == NUM_ACTIONS

    def test_single_segment_edge_case(self, model):
        """All frames have the same predicted label (S=1) must not crash."""
        action_idx = torch.zeros(T, dtype=torch.long)
        batch_input = torch.randn(1, INPUT_DIM, T)
        batch_target = torch.zeros(1, T, dtype=torch.long)
        _, refine_rollout, _ = model(action_idx, batch_input, batch_target)
        assert refine_rollout.shape == (1, NUM_ACTIONS, T)

    def test_gradients_flow(self, model):
        action_idx = _make_segments(T, num_segments=3)
        batch_input = torch.randn(1, INPUT_DIM, T)
        batch_target = torch.randint(0, NUM_ACTIONS, (1, T))
        refine_pred, _, gt_labels = model(action_idx, batch_input, batch_target)
        loss = torch.nn.CrossEntropyLoss()(refine_pred[0], gt_labels.view(-1))
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


# ── train_refiner ─────────────────────────────────────────────────────────────

class TestTrainRefiner:
    @pytest.fixture
    def backbone_dir(self, tmp_path):
        backbone = _DummyBackbone(INPUT_DIM, NUM_ACTIONS)
        bb_dir = tmp_path / "bb_ckpts"
        bb_dir.mkdir()
        torch.save(backbone.state_dict(), str(bb_dir / "epoch-1.model"))
        return backbone, str(bb_dir), [1]

    def test_creates_refiner_checkpoint(self, backbone_dir, tmp_path):
        backbone, bb_dir, bb_epochs = backbone_dir
        refiner = GRURefiner(
            num_actions=NUM_ACTIONS, input_dim=INPUT_DIM, feat_dim=FEAT_DIM,
            num_highlevel_frames=NUM_HL_FRAMES, num_highlevel_samples=NUM_HL_SAMPLES,
            device="cpu",
        )
        refiner_dir = str(tmp_path / "refiner_ckpts")
        bg = MockBatchGenerator(NUM_ACTIONS, INPUT_DIM, T, n_videos=2)

        train_refiner(
            model=refiner, backbone=backbone,
            backbone_model_dir=bb_dir, backbone_epochs=bb_epochs,
            batch_gen=bg, num_epochs=1, learning_rate=1e-3,
            model_dir=refiner_dir, device=torch.device("cpu"),
        )
        assert Path(refiner_dir, "epoch-1.model").exists()
        assert Path(refiner_dir, "epoch-1.opt").exists()

    def test_logs_train_loss_to_wandb(self, backbone_dir, tmp_path):
        backbone, bb_dir, bb_epochs = backbone_dir
        refiner = GRURefiner(
            num_actions=NUM_ACTIONS, input_dim=INPUT_DIM, feat_dim=FEAT_DIM,
            num_highlevel_frames=NUM_HL_FRAMES, num_highlevel_samples=NUM_HL_SAMPLES,
            device="cpu",
        )
        mock_run = MagicMock()
        bg = MockBatchGenerator(NUM_ACTIONS, INPUT_DIM, T, n_videos=2)

        train_refiner(
            model=refiner, backbone=backbone,
            backbone_model_dir=bb_dir, backbone_epochs=bb_epochs,
            batch_gen=bg, num_epochs=1, learning_rate=1e-3,
            model_dir=str(tmp_path / "refiner_ckpts2"),
            device=torch.device("cpu"), wandb_run=mock_run,
        )
        mock_run.log.assert_called()
        assert "train/loss" in mock_run.log.call_args_list[0][0][0]
