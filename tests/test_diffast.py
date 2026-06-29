"""Unit tests for models/DiffAct/model.py — EncoderModel, DecoderModel, and
ASDiffusionModel forward pass, loss computation, and gradient flow.

The full DataLoader-based Trainer.train() is intentionally not tested here
because it requires on-disk VideoFeatureDataset files.  These tests cover the
model internals that are exercised by every training step.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

_BASE = os.path.join(os.path.dirname(__file__), "..")


def _load(unique_name, relpath):
    path = os.path.join(_BASE, relpath)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


_diffact = _load("diffast_model", "models/DiffAct/model.py")

ASDiffusionModel = _diffact.ASDiffusionModel
EncoderModel = _diffact.EncoderModel
DecoderModel = _diffact.DecoderModel

# ── Minimal config for fast tests ─────────────────────────────────────────────

NUM_CLASSES = 4
FEAT_DIM = 16
T = 20           # sequence length (must be > 1 for MSE smoothness terms)
DEVICE = torch.device("cpu")

_ENCODER_PARAMS = {
    "use_instance_norm": False,
    "num_layers": 2,
    "num_f_maps": 8,
    "input_dim": FEAT_DIM,
    "kernel_size": 3,
    "normal_dropout_rate": 0.0,
    "channel_dropout_rate": 0.0,
    "temporal_dropout_rate": 0.0,
    "feature_layer_indices": [0, 1],   # valid indices for 2 layers
}

_DECODER_PARAMS = {
    "num_layers": 2,
    "num_f_maps": 8,
    "time_emb_dim": 32,
    "kernel_size": 3,
    "dropout_rate": 0.0,
    # input_dim is auto-computed by ASDiffusionModel.__init__
}

_DIFFUSION_PARAMS = {
    "timesteps": 50,            # reduced from 1000; only affects schedule
    "sampling_timesteps": 5,
    "ddim_sampling_eta": 1.0,
    "snr_scale": 0.5,
    "cond_types": ["full"],     # deterministic; avoids boundary_gt in forward
    "detach_decoder": False,
}


def _make_model(cond_types=None):
    import copy
    enc = copy.deepcopy(_ENCODER_PARAMS)
    dec = copy.deepcopy(_DECODER_PARAMS)
    diff = copy.deepcopy(_DIFFUSION_PARAMS)
    if cond_types is not None:
        diff["cond_types"] = cond_types
    return ASDiffusionModel(enc, dec, diff, NUM_CLASSES, DEVICE)


def _one_hot(T, num_classes):
    """Create a (1, C, T) one-hot ground-truth tensor."""
    labels = torch.randint(0, num_classes, (T,))
    gt = torch.zeros(1, num_classes, T)
    for t, c in enumerate(labels):
        gt[0, c, t] = 1.0
    return gt


def _boundary_gt(T):
    """Binary (1, 1, T) boundary tensor."""
    bg = torch.zeros(1, 1, T)
    bg[0, 0, T // 2] = 1.0
    return bg


# ── EncoderModel ──────────────────────────────────────────────────────────────

class TestEncoderModel:
    @pytest.fixture
    def model(self):
        enc = dict(_ENCODER_PARAMS)
        enc.pop("use_instance_norm")
        enc["num_classes"] = NUM_CLASSES
        return EncoderModel(**enc)

    def test_forward_output_shape(self, model):
        x = torch.randn(1, FEAT_DIM, T)
        out = model(x, get_features=False)
        assert out.shape == (1, NUM_CLASSES, T)

    def test_forward_with_features_shapes(self, model):
        x = torch.randn(1, FEAT_DIM, T)
        enc_out, backbone_feats = model(x, get_features=True)
        assert enc_out.shape == (1, NUM_CLASSES, T)
        # backbone_feats: len(feature_layer_indices) × num_f_maps concatenated
        expected_dim = len(_ENCODER_PARAMS["feature_layer_indices"]) * _ENCODER_PARAMS["num_f_maps"]
        assert backbone_feats.shape == (1, expected_dim, T)

    def test_forward_output_dtype(self, model):
        x = torch.randn(1, FEAT_DIM, T)
        out = model(x, get_features=False)
        assert out.dtype == torch.float32


# ── DecoderModel ──────────────────────────────────────────────────────────────

class TestDecoderModel:
    @pytest.fixture
    def decoder(self):
        backbone_dim = len(_ENCODER_PARAMS["feature_layer_indices"]) * _ENCODER_PARAMS["num_f_maps"]
        return DecoderModel(
            input_dim=backbone_dim,
            num_classes=NUM_CLASSES,
            num_layers=_DECODER_PARAMS["num_layers"],
            num_f_maps=_DECODER_PARAMS["num_f_maps"],
            time_emb_dim=_DECODER_PARAMS["time_emb_dim"],
            kernel_size=_DECODER_PARAMS["kernel_size"],
            dropout_rate=_DECODER_PARAMS["dropout_rate"],
        )

    def test_forward_output_shape(self, decoder):
        backbone_dim = len(_ENCODER_PARAMS["feature_layer_indices"]) * _ENCODER_PARAMS["num_f_maps"]
        backbone_feats = torch.randn(1, backbone_dim, T)
        t = torch.tensor([5], dtype=torch.long)
        event = torch.rand(1, NUM_CLASSES, T)  # [0,1] — treated as diffused probs
        out = decoder(backbone_feats, t, event)
        assert out.shape == (1, NUM_CLASSES, T)


# ── ASDiffusionModel ──────────────────────────────────────────────────────────

class TestASDiffusionModel:
    def test_instantiation_succeeds(self):
        model = _make_model()
        assert model is not None

    def test_ddim_sample_output_shape(self):
        model = _make_model()
        model.eval()
        video_feats = torch.randn(1, FEAT_DIM, T)
        with torch.no_grad():
            out = model.ddim_sample(video_feats)
        assert out.shape == (1, NUM_CLASSES, T)

    def test_ddim_sample_values_in_unit_range(self):
        model = _make_model()
        model.eval()
        video_feats = torch.randn(1, FEAT_DIM, T)
        with torch.no_grad():
            out = model.ddim_sample(video_feats)
        # ddim_sample returns denormalized probabilities in [0, 1]
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_prepare_targets_shapes(self):
        model = _make_model()
        event_gt = _one_hot(T, NUM_CLASSES)
        event_diffused, noise, t = model.prepare_targets(event_gt)
        assert event_diffused.shape == event_gt.shape
        assert noise.shape == event_gt.shape
        assert t.shape == (1,)

    def test_prepare_targets_event_diffused_in_range(self):
        model = _make_model()
        event_gt = _one_hot(T, NUM_CLASSES)
        event_diffused, _, _ = model.prepare_targets(event_gt)
        assert event_diffused.min() >= 0.0
        assert event_diffused.max() <= 1.0

    def test_get_training_loss_returns_all_keys(self):
        model = _make_model()
        model.train()
        event_gt = _one_hot(T, NUM_CLASSES)
        bg = _boundary_gt(T)
        loss_dict = model.get_training_loss(
            video_feats=torch.randn(1, FEAT_DIM, T),
            event_gt=event_gt,
            boundary_gt=bg,
            encoder_ce_criterion=nn.CrossEntropyLoss(),
            encoder_mse_criterion=nn.MSELoss(reduction="none"),
            encoder_boundary_criterion=nn.MSELoss(reduction="none"),
            decoder_ce_criterion=nn.CrossEntropyLoss(),
            decoder_mse_criterion=nn.MSELoss(reduction="none"),
            decoder_boundary_criterion=nn.MSELoss(reduction="none"),
            soft_label=None,
        )
        expected_keys = {
            "encoder_ce_loss", "encoder_mse_loss", "encoder_boundary_loss",
            "decoder_ce_loss", "decoder_mse_loss", "decoder_boundary_loss",
        }
        assert set(loss_dict.keys()) == expected_keys

    def test_get_training_loss_values_are_finite(self):
        model = _make_model()
        model.train()
        event_gt = _one_hot(T, NUM_CLASSES)
        bg = _boundary_gt(T)
        loss_dict = model.get_training_loss(
            video_feats=torch.randn(1, FEAT_DIM, T),
            event_gt=event_gt,
            boundary_gt=bg,
            encoder_ce_criterion=nn.CrossEntropyLoss(),
            encoder_mse_criterion=nn.MSELoss(reduction="none"),
            encoder_boundary_criterion=nn.MSELoss(reduction="none"),
            decoder_ce_criterion=nn.CrossEntropyLoss(),
            decoder_mse_criterion=nn.MSELoss(reduction="none"),
            decoder_boundary_criterion=nn.MSELoss(reduction="none"),
            soft_label=None,
        )
        for name, val in loss_dict.items():
            assert torch.isfinite(val).all(), f"{name} is not finite"

    def test_gradients_flow_through_training_loss(self):
        model = _make_model()
        model.train()
        event_gt = _one_hot(T, NUM_CLASSES)
        bg = _boundary_gt(T)
        loss_dict = model.get_training_loss(
            video_feats=torch.randn(1, FEAT_DIM, T),
            event_gt=event_gt,
            boundary_gt=bg,
            encoder_ce_criterion=nn.CrossEntropyLoss(),
            encoder_mse_criterion=nn.MSELoss(reduction="none"),
            encoder_boundary_criterion=nn.MSELoss(reduction="none"),
            decoder_ce_criterion=nn.CrossEntropyLoss(),
            decoder_mse_criterion=nn.MSELoss(reduction="none"),
            decoder_boundary_criterion=nn.MSELoss(reduction="none"),
            soft_label=None,
        )
        total = sum(v for v in loss_dict.values() if v.requires_grad)
        total.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_encoder_boundary_loss_is_zero(self):
        """Encoder boundary loss is always zero (hardcoded in the model)."""
        model = _make_model()
        model.train()
        event_gt = _one_hot(T, NUM_CLASSES)
        bg = _boundary_gt(T)
        loss_dict = model.get_training_loss(
            video_feats=torch.randn(1, FEAT_DIM, T),
            event_gt=event_gt,
            boundary_gt=bg,
            encoder_ce_criterion=nn.CrossEntropyLoss(),
            encoder_mse_criterion=nn.MSELoss(reduction="none"),
            encoder_boundary_criterion=nn.MSELoss(reduction="none"),
            decoder_ce_criterion=nn.CrossEntropyLoss(),
            decoder_mse_criterion=nn.MSELoss(reduction="none"),
            decoder_boundary_criterion=nn.MSELoss(reduction="none"),
            soft_label=None,
        )
        assert loss_dict["encoder_boundary_loss"].item() == 0.0

    def test_cond_type_zero_still_produces_valid_output(self):
        """'zero' cond_type feeds zeros as backbone features — must not crash."""
        model = _make_model(cond_types=["zero"])
        model.train()
        event_gt = _one_hot(T, NUM_CLASSES)
        bg = _boundary_gt(T)
        loss_dict = model.get_training_loss(
            video_feats=torch.randn(1, FEAT_DIM, T),
            event_gt=event_gt,
            boundary_gt=bg,
            encoder_ce_criterion=nn.CrossEntropyLoss(),
            encoder_mse_criterion=nn.MSELoss(reduction="none"),
            encoder_boundary_criterion=nn.MSELoss(reduction="none"),
            decoder_ce_criterion=nn.CrossEntropyLoss(),
            decoder_mse_criterion=nn.MSELoss(reduction="none"),
            decoder_boundary_criterion=nn.MSELoss(reduction="none"),
            soft_label=None,
        )
        for val in loss_dict.values():
            assert torch.isfinite(val).all()
