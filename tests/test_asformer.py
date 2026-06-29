"""Unit tests for models/ASFormer/ — MyTransformer forward pass, BatchGenerator
interface, and the Trainer training + evaluation + logging loop.

Key differences from mstcn:
  - BatchGenerator.next_batch() returns 4-tuple (input, target, mask, vid_names)
  - AttLayer._sliding_window_self_att asserts batch_size == 1; all tests use B=1
  - Checkpoints are written only at epoch % 10 == 0 when batch_gen_tst is provided
  - Module-level `device` is resolved at import time (cpu in Docker/CI)
"""

import importlib.util
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


# ASFormer's batch_gen imports grid_sampler from the same directory; add the
# directory to sys.path only for loading that one module, then remove it.
_asf_dir = os.path.join(_BASE, "models", "ASFormer")
sys.path.insert(0, _asf_dir)
_asformer = _load("asformer_model", "models/ASFormer/model.py")
sys.path.pop(0)

# ASFormer uses a module-level `device` variable for window_mask creation and
# for placing tensors in _sliding_window_self_att.  Override it to CPU so tests
# run without a GPU (and without device mismatch when inputs are CPU tensors).
_asformer.device = torch.device("cpu")

MyTransformer = _asformer.MyTransformer
Trainer = _asformer.Trainer

# ── Hyperparameters for fast tests ────────────────────────────────────────────

NUM_CLASSES = 3
FEAT_DIM = 16
NUM_LAYERS = 4
NUM_F_MAPS = 16
SEQ_LEN = 48          # must be >= 2 * bl (bl = NUM_F_MAPS default) for sliding att
CHANNEL_MASK_RATE = 0.0


# ── Mock batch generator with ASFormer's 4-tuple return ───────────────────────

class MockASFormerBatchGen:
    """Returns (input, target, mask, vid_names) — matches ASFormer's interface."""

    def __init__(self, num_classes, feat_dim=16, seq_len=48, n_videos=2):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.list_of_examples = [f"vid{i}" for i in range(n_videos)]
        self.index = 0

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def reset(self):
        self.index = 0

    def next_batch(self, batch_size, if_warp=False):
        batch = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size
        B = len(batch)
        inp = torch.randn(B, self.feat_dim, self.seq_len)
        tgt = torch.randint(0, self.num_classes, (B, self.seq_len))
        mask = torch.ones(B, self.num_classes, self.seq_len)
        return inp, tgt, mask, batch


# ── MyTransformer ─────────────────────────────────────────────────────────────

class TestMyTransformer:
    @pytest.fixture
    def model(self):
        return MyTransformer(
            num_decoders=2,
            num_layers=NUM_LAYERS,
            r1=2, r2=2,
            num_f_maps=NUM_F_MAPS,
            input_dim=FEAT_DIM,
            num_classes=NUM_CLASSES,
            channel_masking_rate=CHANNEL_MASK_RATE,
        )

    def test_output_shape(self, model):
        # (1 encoder + num_decoders, B, C, T)
        x = torch.randn(1, FEAT_DIM, SEQ_LEN)
        mask = torch.ones(1, NUM_CLASSES, SEQ_LEN)
        out = model(x, mask)
        assert out.shape == (1 + 2, 1, NUM_CLASSES, SEQ_LEN)

    def test_output_dtype_float(self, model):
        x = torch.randn(1, FEAT_DIM, SEQ_LEN)
        mask = torch.ones(1, NUM_CLASSES, SEQ_LEN)
        out = model(x, mask)
        assert out.dtype == torch.float32

    def test_mask_zeros_outputs_zeros(self, model):
        model.eval()
        x = torch.randn(1, FEAT_DIM, SEQ_LEN)
        mask = torch.zeros(1, NUM_CLASSES, SEQ_LEN)
        with torch.no_grad():
            out = model(x, mask)
        assert torch.all(out == 0.0)

    def test_trainer_hardcodes_three_decoders(self):
        """Trainer always builds MyTransformer(3, ...) → output has 4 stages."""
        trainer = Trainer(NUM_LAYERS, 2, 2, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES, CHANNEL_MASK_RATE)
        x = torch.randn(1, FEAT_DIM, SEQ_LEN)
        mask = torch.ones(1, NUM_CLASSES, SEQ_LEN)
        trainer.model.eval()
        with torch.no_grad():
            out = trainer.model(x, mask)
        assert out.shape[0] == 4  # 1 encoder + 3 decoders


# ── Trainer ───────────────────────────────────────────────────────────────────

class TestASFormerTrainer:
    @pytest.fixture
    def trainer(self):
        return Trainer(NUM_LAYERS, 2, 2, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES, CHANNEL_MASK_RATE)

    def test_train_logs_loss_to_wandb(self, trainer, tmp_path, device):
        bg = MockASFormerBatchGen(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        mock_run = MagicMock()
        trainer.train(
            str(tmp_path), bg, num_epochs=1, batch_size=1,
            learning_rate=1e-3, wandb_run=mock_run,
        )
        mock_run.log.assert_called()
        logged = mock_run.log.call_args_list[0][0][0]
        assert "train/loss" in logged

    def test_train_logs_acc_to_wandb(self, trainer, tmp_path, device):
        bg = MockASFormerBatchGen(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        mock_run = MagicMock()
        trainer.train(
            str(tmp_path), bg, num_epochs=1, batch_size=1,
            learning_rate=1e-3, wandb_run=mock_run,
        )
        logged = mock_run.log.call_args_list[0][0][0]
        assert "train/acc" in logged

    def test_train_creates_checkpoint_at_epoch_10(self, trainer, tmp_path, device):
        """Checkpoints are saved every 10 epochs when batch_gen_tst is provided."""
        bg = MockASFormerBatchGen(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        bg_tst = MockASFormerBatchGen(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer.train(
            str(tmp_path), bg, num_epochs=10, batch_size=1,
            learning_rate=1e-3, batch_gen_tst=bg_tst,
        )
        assert (tmp_path / "epoch-10.model").exists()
        assert (tmp_path / "epoch-10.opt").exists()

    def test_train_no_checkpoint_without_test_gen(self, trainer, tmp_path, device):
        """Without batch_gen_tst, no checkpoints are written at all."""
        bg = MockASFormerBatchGen(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer.train(
            str(tmp_path), bg, num_epochs=10, batch_size=1,
            learning_rate=1e-3, batch_gen_tst=None,
        )
        model_files = list(tmp_path.glob("*.model"))
        assert len(model_files) == 0

    def test_train_logs_val_metrics_with_eval_data(
        self, trainer, tmp_path, make_dataset, actions_dict, device
    ):
        feat_path, gt_path, bundle = make_dataset(["asf_vid"], FEAT_DIM, SEQ_LEN)
        eval_data = (feat_path, bundle, gt_path, actions_dict, 1)

        bg = MockASFormerBatchGen(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        mock_run = MagicMock()
        trainer.train(
            str(tmp_path), bg, num_epochs=1, batch_size=1,
            learning_rate=1e-3, wandb_run=mock_run, eval_data=eval_data,
        )
        all_keys = set()
        for c in mock_run.log.call_args_list:
            all_keys.update(c[0][0].keys())
        assert "val/acc" in all_keys
        assert "val/edit" in all_keys

    def test_evaluate_returns_all_metric_keys(
        self, trainer, make_dataset, actions_dict, device
    ):
        feat_path, gt_path, bundle = make_dataset(["v1", "v2"], FEAT_DIM, SEQ_LEN)
        metrics = trainer.evaluate(feat_path, bundle, gt_path, actions_dict, 1, device)
        assert set(metrics.keys()) == {
            "val/acc", "val/edit", "val/f1_10", "val/f1_25", "val/f1_50", "val/dr"
        }

    def test_evaluate_accuracy_in_valid_range(
        self, trainer, make_dataset, actions_dict, device
    ):
        feat_path, gt_path, bundle = make_dataset(["v1"], FEAT_DIM, SEQ_LEN)
        metrics = trainer.evaluate(feat_path, bundle, gt_path, actions_dict, 1, device)
        assert 0.0 <= metrics["val/acc"] <= 100.0

    def test_saved_checkpoint_loads_correctly(self, trainer, tmp_path, device):
        bg = MockASFormerBatchGen(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        bg_tst = MockASFormerBatchGen(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer.train(
            str(tmp_path), bg, num_epochs=10, batch_size=1,
            learning_rate=1e-3, batch_gen_tst=bg_tst,
        )
        model2 = MyTransformer(
            num_decoders=3, num_layers=NUM_LAYERS, r1=2, r2=2,
            num_f_maps=NUM_F_MAPS, input_dim=FEAT_DIM,
            num_classes=NUM_CLASSES, channel_masking_rate=CHANNEL_MASK_RATE,
        )
        model2.load_state_dict(
            torch.load(str(tmp_path / "epoch-10.model"), map_location=device)
        )
