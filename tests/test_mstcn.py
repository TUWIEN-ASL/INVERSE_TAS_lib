"""Unit tests for models/mstcn/ — MultiStageModel, SingleStageModel, BatchGenerator,
and the Trainer training + evaluation + logging loop."""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Project root already added by conftest.py; helpers.py is in the same directory
from helpers import MockBatchGenerator

_BASE = os.path.join(os.path.dirname(__file__), "..")


def _load(unique_name, relpath):
    """Load a project module by its file path, registering under a unique name."""
    path = os.path.join(_BASE, relpath)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mstcn = _load("mstcn_model", "models/mstcn/model.py")
_bg_mod = _load("mstcn_batch_gen", "models/mstcn/batch_gen.py")

MultiStageModel = _mstcn.MultiStageModel
SingleStageModel = _mstcn.SingleStageModel
Trainer = _mstcn.Trainer
BatchGenerator = _bg_mod.BatchGenerator

# ── Hyperparameter defaults for fast tests ────────────────────────────────────

NUM_CLASSES = 3
FEAT_DIM = 16
NUM_LAYERS = 4
NUM_F_MAPS = 16
NUM_STAGES = 2
SEQ_LEN = 48


# ── MultiStageModel ───────────────────────────────────────────────────────────

class TestMultiStageModel:
    def test_output_shape(self):
        model = MultiStageModel(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        x = torch.randn(1, FEAT_DIM, SEQ_LEN)
        mask = torch.ones(1, NUM_CLASSES, SEQ_LEN)
        out = model(x, mask)
        assert out.shape == (NUM_STAGES, 1, NUM_CLASSES, SEQ_LEN)

    def test_output_zeros_when_mask_is_zero(self):
        model = MultiStageModel(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        model.eval()
        x = torch.randn(1, FEAT_DIM, SEQ_LEN)
        mask = torch.zeros(1, NUM_CLASSES, SEQ_LEN)
        with torch.no_grad():
            out = model(x, mask)
        assert torch.all(out == 0.0)

    def test_batch_size_gt1(self):
        model = MultiStageModel(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        x = torch.randn(3, FEAT_DIM, SEQ_LEN)
        mask = torch.ones(3, NUM_CLASSES, SEQ_LEN)
        out = model(x, mask)
        assert out.shape[1] == 3


# ── SingleStageModel ──────────────────────────────────────────────────────────

class TestSingleStageModel:
    def test_output_shape(self):
        model = SingleStageModel(NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        x = torch.randn(1, FEAT_DIM, SEQ_LEN)
        mask = torch.ones(1, NUM_CLASSES, SEQ_LEN)
        out = model(x, mask)
        assert out.shape == (1, NUM_CLASSES, SEQ_LEN)

    def test_output_dtype(self):
        model = SingleStageModel(NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        x = torch.randn(1, FEAT_DIM, SEQ_LEN)
        mask = torch.ones(1, NUM_CLASSES, SEQ_LEN)
        out = model(x, mask)
        assert out.dtype == torch.float32


# ── BatchGenerator ────────────────────────────────────────────────────────────

class TestBatchGenerator:
    @pytest.fixture
    def dataset(self, tmp_path, actions_dict):
        feat_dir = tmp_path / "features"
        gt_dir = tmp_path / "annotations"
        feat_dir.mkdir()
        gt_dir.mkdir()

        label_list = list(actions_dict.keys())
        for vid in ["vid1", "vid2", "vid3"]:
            feats = np.random.randn(SEQ_LEN, FEAT_DIM).astype(np.float32)
            np.save(str(feat_dir / f"{vid}.npy"), feats)
            labels = [label_list[i % NUM_CLASSES] for i in range(SEQ_LEN)]
            (gt_dir / f"{vid}.txt").write_text(
                "### Frame level recognition: ###\n" + "\n".join(labels) + "\n"
            )

        bundle = tmp_path / "train.bundle"
        bundle.write_text("vid1\nvid2\nvid3\n")
        return str(feat_dir) + "/", str(gt_dir) + "/", str(bundle)

    def test_read_data_populates_list(self, dataset, actions_dict):
        feat_path, gt_path, bundle = dataset
        bg = BatchGenerator(NUM_CLASSES, actions_dict, gt_path, feat_path, sample_rate=1)
        bg.read_data(bundle)
        assert len(bg.list_of_examples) == 3

    def test_has_next_true_after_read(self, dataset, actions_dict):
        feat_path, gt_path, bundle = dataset
        bg = BatchGenerator(NUM_CLASSES, actions_dict, gt_path, feat_path, sample_rate=1)
        bg.read_data(bundle)
        assert bg.has_next()

    def test_has_next_false_after_exhaustion(self, dataset, actions_dict):
        feat_path, gt_path, bundle = dataset
        bg = BatchGenerator(NUM_CLASSES, actions_dict, gt_path, feat_path, sample_rate=1)
        bg.read_data(bundle)
        while bg.has_next():
            bg.next_batch(1)
        assert not bg.has_next()

    def test_next_batch_shapes(self, dataset, actions_dict):
        feat_path, gt_path, bundle = dataset
        bg = BatchGenerator(NUM_CLASSES, actions_dict, gt_path, feat_path, sample_rate=1)
        bg.read_data(bundle)
        inp, tgt, mask = bg.next_batch(1)
        assert inp.shape[0] == 1
        assert inp.shape[1] == FEAT_DIM
        assert tgt.shape[0] == 1
        assert mask.shape[:2] == (1, NUM_CLASSES)

    def test_reset_allows_reiteration(self, dataset, actions_dict):
        feat_path, gt_path, bundle = dataset
        bg = BatchGenerator(NUM_CLASSES, actions_dict, gt_path, feat_path, sample_rate=1)
        bg.read_data(bundle)
        while bg.has_next():
            bg.next_batch(1)
        bg.reset()
        assert bg.has_next()

    def test_sample_rate_reduces_seq_len(self, dataset, actions_dict):
        feat_path, gt_path, bundle = dataset
        bg1 = BatchGenerator(NUM_CLASSES, actions_dict, gt_path, feat_path, sample_rate=1)
        bg2 = BatchGenerator(NUM_CLASSES, actions_dict, gt_path, feat_path, sample_rate=2)
        bg1.read_data(bundle)
        bg2.read_data(bundle)
        inp1, _, _ = bg1.next_batch(1)
        inp2, _, _ = bg2.next_batch(1)
        assert inp2.shape[2] <= inp1.shape[2]


# ── Trainer ───────────────────────────────────────────────────────────────────

class TestTrainer:
    def test_train_creates_model_checkpoint(self, tmp_path, device):
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer = Trainer(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device)
        assert (tmp_path / "epoch-1.model").exists()

    def test_train_creates_optimizer_checkpoint(self, tmp_path, device):
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer = Trainer(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device)
        assert (tmp_path / "epoch-1.opt").exists()

    def test_train_multiple_epochs_saves_each(self, tmp_path, device):
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer = Trainer(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        trainer.train(str(tmp_path), bg, num_epochs=3, batch_size=1,
                      learning_rate=1e-3, device=device)
        for ep in range(1, 4):
            assert (tmp_path / f"epoch-{ep}.model").exists()
            assert (tmp_path / f"epoch-{ep}.opt").exists()

    def test_train_logs_loss_and_acc_to_wandb(self, tmp_path, device):
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        mock_run = MagicMock()
        trainer = Trainer(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device, wandb_run=mock_run)
        mock_run.log.assert_called()
        logged = mock_run.log.call_args_list[0][0][0]
        assert "train/loss" in logged
        assert "train/acc" in logged

    def test_train_logs_val_metrics_with_eval_data(
        self, tmp_path, make_dataset, actions_dict, device
    ):
        feat_path, gt_path, bundle = make_dataset(["eval_vid"], FEAT_DIM, SEQ_LEN)
        eval_data = (feat_path, bundle, gt_path, actions_dict, 1)

        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        mock_run = MagicMock()
        trainer = Trainer(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device,
                      wandb_run=mock_run, eval_data=eval_data)

        all_keys = set()
        for c in mock_run.log.call_args_list:
            all_keys.update(c[0][0].keys())
        assert "val/acc" in all_keys
        assert "val/edit" in all_keys

    def test_evaluate_returns_all_metric_keys(self, make_dataset, actions_dict, device):
        feat_path, gt_path, bundle = make_dataset(["v1", "v2"], FEAT_DIM, SEQ_LEN)
        trainer = Trainer(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        metrics = trainer.evaluate(feat_path, bundle, gt_path, actions_dict, 1, device)
        assert set(metrics.keys()) == {
            "val/acc", "val/edit", "val/f1_10", "val/f1_25", "val/f1_50", "val/dr"
        }

    def test_evaluate_accuracy_in_valid_range(self, make_dataset, actions_dict, device):
        feat_path, gt_path, bundle = make_dataset(["v1"], FEAT_DIM, SEQ_LEN)
        trainer = Trainer(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        metrics = trainer.evaluate(feat_path, bundle, gt_path, actions_dict, 1, device)
        assert 0.0 <= metrics["val/acc"] <= 100.0

    def test_saved_checkpoint_loads_correctly(self, tmp_path, device):
        bg = MockBatchGenerator(NUM_CLASSES, FEAT_DIM, SEQ_LEN, n_videos=2)
        trainer = Trainer(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        trainer.train(str(tmp_path), bg, num_epochs=1, batch_size=1,
                      learning_rate=1e-3, device=device)
        model2 = MultiStageModel(NUM_STAGES, NUM_LAYERS, NUM_F_MAPS, FEAT_DIM, NUM_CLASSES)
        model2.load_state_dict(
            torch.load(str(tmp_path / "epoch-1.model"), map_location=device)
        )
