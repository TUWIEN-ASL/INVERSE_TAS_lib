"""Unit tests for models/onlinetas/ — OnlineTASModel, MemoryBank, Trainer,
_split_into_clips, and the post_process post-processing step."""

import importlib.util
import math
import os
import sys
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


_ot = _load("onlinetas_model", "models/onlinetas/model.py")

MemoryBank = _ot.MemoryBank
OnlineTASModel = _ot.OnlineTASModel
Trainer = _ot.Trainer
post_process = _ot.post_process

# ── Hyperparameter defaults for fast tests ────────────────────────────────────

NUM_CLASSES = 3
FEAT_DIM = 16
NUM_LAYERS = 4
NUM_F_MAPS = 16
CLIP_WIDTH = 8    # small w keeps tests fast
SEQ_LEN = 32     # exact multiple of CLIP_WIDTH


def _make_trainer(**kwargs):
    defaults = dict(
        input_dim=FEAT_DIM, num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS, num_f_maps=NUM_F_MAPS, w=CLIP_WIDTH,
        num_iterations=1, num_td_layers=1, td_heads=4, sa_heads=4,
    )
    defaults.update(kwargs)
    return Trainer(**defaults)


def _make_model(**kwargs):
    defaults = dict(
        input_dim=FEAT_DIM, num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS, num_f_maps=NUM_F_MAPS, w=CLIP_WIDTH,
        num_iterations=1, num_td_layers=1, td_heads=4, sa_heads=4,
    )
    defaults.update(kwargs)
    return OnlineTASModel(**defaults)


# ── OnlineTASModel ────────────────────────────────────────────────────────────

class TestOnlineTASModel:
    def test_logits_shape(self):
        model = _make_model()
        clip = torch.randn(1, FEAT_DIM, CLIP_WIDTH)
        M = torch.zeros(1, FEAT_DIM, CLIP_WIDTH)
        logits, _ = model.forward_clip(clip, M)
        assert logits.shape == (1, NUM_CLASSES, CLIP_WIDTH)

    def test_c_tilde_shape(self):
        model = _make_model()
        clip = torch.randn(1, FEAT_DIM, CLIP_WIDTH)
        M = torch.zeros(1, FEAT_DIM, CLIP_WIDTH)
        _, c_tilde = model.forward_clip(clip, M)
        assert c_tilde.shape == (1, FEAT_DIM, CLIP_WIDTH)

    def test_batch_size_gt1(self):
        model = _make_model()
        clip = torch.randn(2, FEAT_DIM, CLIP_WIDTH)
        M = torch.zeros(2, FEAT_DIM, CLIP_WIDTH)
        logits, _ = model.forward_clip(clip, M)
        assert logits.shape[0] == 2


# ── MemoryBank ────────────────────────────────────────────────────────────────

class TestMemoryBank:
    def test_initialize_creates_empty_m_long(self):
        bank = MemoryBank(CLIP_WIDTH, FEAT_DIM, torch.device("cpu"))
        bank.initialize(torch.randn(1, FEAT_DIM, CLIP_WIDTH))
        assert bank.M_long.shape == (1, FEAT_DIM, 0)

    def test_initialize_stores_prev_c_tilde(self):
        bank = MemoryBank(CLIP_WIDTH, FEAT_DIM, torch.device("cpu"))
        bank.initialize(torch.randn(1, FEAT_DIM, CLIP_WIDTH))
        assert bank.prev_c_tilde.shape == (1, FEAT_DIM, CLIP_WIDTH)

    def test_get_initial_returns_full_width(self):
        bank = MemoryBank(CLIP_WIDTH, FEAT_DIM, torch.device("cpu"))
        bank.initialize(torch.randn(1, FEAT_DIM, CLIP_WIDTH))
        assert bank.get().shape == (1, FEAT_DIM, CLIP_WIDTH)

    def test_update_appends_compressed_token(self):
        bank = MemoryBank(CLIP_WIDTH, FEAT_DIM, torch.device("cpu"))
        bank.initialize(torch.zeros(1, FEAT_DIM, CLIP_WIDTH))
        bank.update(torch.zeros(1, FEAT_DIM, CLIP_WIDTH), torch.zeros(1, FEAT_DIM, 1))
        assert bank.M_long.shape[-1] == 1

    def test_get_after_update_preserves_width(self):
        bank = MemoryBank(CLIP_WIDTH, FEAT_DIM, torch.device("cpu"))
        bank.initialize(torch.zeros(1, FEAT_DIM, CLIP_WIDTH))
        bank.update(torch.zeros(1, FEAT_DIM, CLIP_WIDTH), torch.zeros(1, FEAT_DIM, 1))
        assert bank.get().shape == (1, FEAT_DIM, CLIP_WIDTH)

    def test_m_long_fifo_at_capacity(self):
        max_long = int(math.floor(2.0 / 3.0 * CLIP_WIDTH))
        bank = MemoryBank(CLIP_WIDTH, FEAT_DIM, torch.device("cpu"))
        bank.initialize(torch.zeros(1, FEAT_DIM, CLIP_WIDTH))
        for _ in range(max_long):
            bank.update(torch.zeros(1, FEAT_DIM, CLIP_WIDTH), torch.zeros(1, FEAT_DIM, 1))
        assert bank.M_long.shape[-1] == max_long

        bank.update(torch.ones(1, FEAT_DIM, CLIP_WIDTH), torch.ones(1, FEAT_DIM, 1))
        assert bank.M_long.shape[-1] == max_long


# ── post_process ──────────────────────────────────────────────────────────────

class TestPostProcess:
    def test_no_change_when_all_confident(self):
        preds = np.array([0, 0, 1, 1, 2])
        confs = np.full(5, 0.99)
        result = post_process(preds, confs, theta=0.5, l_min=3)
        np.testing.assert_array_equal(result, preds)

    def test_output_same_length_as_input(self):
        preds = np.array([0, 1, 2, 1, 0])
        confs = np.full(5, 0.9)
        result = post_process(preds, confs, theta=0.5, l_min=2)
        assert len(result) == len(preds)

    def test_low_confidence_short_segment_replaced(self):
        preds = np.array([0, 0, 1, 0, 0])
        confs = np.array([0.99, 0.99, 0.1, 0.99, 0.99])
        result = post_process(preds, confs, theta=0.5, l_min=4)
        assert result[2] == result[1]


# ── Trainer._split_into_clips ─────────────────────────────────────────────────

class TestSplitIntoClips:
    @pytest.fixture
    def trainer(self):
        return _make_trainer()

    def test_exact_multiple_no_padding(self, trainer):
        T = CLIP_WIDTH * 4
        features = torch.randn(1, FEAT_DIM, T)
        labels = torch.zeros(1, T, dtype=torch.long)
        mask = torch.ones(1, NUM_CLASSES, T)
        clips, _, _ = trainer._split_into_clips(features, labels, mask)
        assert len(clips) == 4
        for c in clips:
            assert c.shape[2] == CLIP_WIDTH

    def test_remainder_padded_to_clip_width(self, trainer):
        T = CLIP_WIDTH * 2 + 3
        features = torch.randn(1, FEAT_DIM, T)
        labels = torch.full((1, T), -100, dtype=torch.long)
        mask = torch.ones(1, NUM_CLASSES, T)
        clips, _, _ = trainer._split_into_clips(features, labels, mask)
        assert clips[-1].shape[2] == CLIP_WIDTH

    def test_correct_number_of_clips(self, trainer):
        import math
        T = CLIP_WIDTH * 3 + 1
        features = torch.randn(1, FEAT_DIM, T)
        labels = torch.zeros(1, T, dtype=torch.long)
        mask = torch.ones(1, NUM_CLASSES, T)
        clips, _, _ = trainer._split_into_clips(features, labels, mask)
        assert len(clips) == math.ceil(T / CLIP_WIDTH)


# ── Trainer ───────────────────────────────────────────────────────────────────

class TestOnlineTASTrainer:
    def test_train_creates_model_checkpoint(self, tmp_path, device):
        trainer = _make_trainer()
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device)
        assert (tmp_path / "epoch-1.model").exists()

    def test_train_creates_optimizer_checkpoint(self, tmp_path, device):
        trainer = _make_trainer()
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device)
        assert (tmp_path / "epoch-1.opt").exists()

    def test_train_logs_loss_and_acc_to_wandb(self, tmp_path, device):
        trainer = _make_trainer()
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        mock_run = MagicMock()
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device, wandb_run=mock_run)
        mock_run.log.assert_called()
        logged = mock_run.log.call_args_list[0][0][0]
        assert "train/loss" in logged
        assert "train/acc" in logged

    def test_train_logs_val_metrics_with_eval_data(
        self, tmp_path, make_dataset, actions_dict, device
    ):
        feat_path, gt_path, bundle = make_dataset(["ot_vid1"], FEAT_DIM, SEQ_LEN)
        eval_data = (feat_path, bundle, gt_path, actions_dict, 1)
        trainer = _make_trainer()
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        mock_run = MagicMock()
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device,
                      wandb_run=mock_run, eval_data=eval_data)
        all_keys = set()
        for c in mock_run.log.call_args_list:
            all_keys.update(c[0][0].keys())
        assert "val/acc" in all_keys

    def test_saved_checkpoint_loads_correctly(self, tmp_path, device):
        trainer = _make_trainer()
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device)
        model2 = _make_model()
        model2.load_state_dict(
            torch.load(str(tmp_path / "epoch-1.model"), map_location=device)
        )
