"""Unit tests for models/video_features/models/transforms.py.

All transforms are pure tensor/PIL operations with no model loading or GPU
dependency.  Tests verify output shape, dtype, and value correctness.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

_BASE = os.path.join(os.path.dirname(__file__), "..")


def _load(unique_name, relpath):
    path = os.path.join(_BASE, relpath)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


_tr = _load("vf_transforms", "models/video_features/models/transforms.py")

TensorCenterCrop  = _tr.TensorCenterCrop
ScaleTo1_1        = _tr.ScaleTo1_1
PermuteAndUnsqueeze = _tr.PermuteAndUnsqueeze
Clamp             = _tr.Clamp
ToUInt8           = _tr.ToUInt8
PILToTensor       = _tr.PILToTensor
ToFloat           = _tr.ToFloat
ResizeImproved    = _tr.ResizeImproved
ToCFHW_ToFloat    = _tr.ToCFHW_ToFloat
ToFCHW            = _tr.ToFCHW


# ── TensorCenterCrop ──────────────────────────────────────────────────────────

class TestTensorCenterCrop:
    def test_output_shape(self):
        t = TensorCenterCrop(crop_size=4)
        x = torch.randn(1, 3, 10, 10)
        assert t(x).shape == (1, 3, 4, 4)

    def test_crop_size_matches_when_equal(self):
        t = TensorCenterCrop(crop_size=8)
        x = torch.randn(2, 3, 8, 8)
        assert t(x).shape == (2, 3, 8, 8)

    def test_crop_is_centered(self):
        """Cropping a constant-value tensor leaves all values unchanged."""
        t = TensorCenterCrop(crop_size=4)
        x = torch.ones(1, 3, 8, 8)
        result = t(x)
        assert torch.all(result == 1.0)

    def test_batched_input(self):
        t = TensorCenterCrop(crop_size=6)
        x = torch.randn(4, 3, 12, 12)
        assert t(x).shape == (4, 3, 6, 6)


# ── ScaleTo1_1 ────────────────────────────────────────────────────────────────

class TestScaleTo1_1:
    def test_zero_maps_to_minus_one(self):
        t = ScaleTo1_1()
        x = torch.zeros(1)
        assert t(x).item() == pytest.approx(-1.0)

    def test_255_maps_to_one(self):
        t = ScaleTo1_1()
        x = torch.full((1,), 255.0)
        assert t(x).item() == pytest.approx(1.0)

    def test_output_shape_preserved(self):
        t = ScaleTo1_1()
        x = torch.randint(0, 255, (2, 3, 4, 4)).float()
        assert t(x).shape == x.shape

    def test_values_in_range(self):
        t = ScaleTo1_1()
        x = torch.randint(0, 256, (10, 10)).float()
        out = t(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


# ── PermuteAndUnsqueeze ───────────────────────────────────────────────────────

class TestPermuteAndUnsqueeze:
    def test_output_shape(self):
        # (T, C, H, W) → (1, C, T, H, W)
        t = PermuteAndUnsqueeze()
        x = torch.randn(8, 3, 16, 16)    # (T, C, H, W)
        out = t(x)
        assert out.shape == (1, 3, 8, 16, 16)

    def test_adds_batch_dimension(self):
        t = PermuteAndUnsqueeze()
        x = torch.randn(4, 2, 6, 6)
        assert t(x).shape[0] == 1

    def test_channel_dim_moves_to_position_1(self):
        t = PermuteAndUnsqueeze()
        T, C, H, W = 5, 7, 8, 8
        x = torch.randn(T, C, H, W)
        out = t(x)
        assert out.shape == (1, C, T, H, W)


# ── Clamp ─────────────────────────────────────────────────────────────────────

class TestClamp:
    def test_values_clamped_to_range(self):
        t = Clamp(-10, 10)
        x = torch.tensor([-20.0, 0.0, 20.0])
        out = t(x)
        assert out[0].item() == pytest.approx(-10.0)
        assert out[1].item() == pytest.approx(0.0)
        assert out[2].item() == pytest.approx(10.0)

    def test_output_shape_unchanged(self):
        t = Clamp(-5, 5)
        x = torch.randn(3, 4, 5)
        assert t(x).shape == x.shape

    def test_values_within_range_unchanged(self):
        t = Clamp(-100, 100)
        x = torch.tensor([1.0, 2.0, -3.0])
        assert torch.allclose(t(x), x)


# ── ToUInt8 ───────────────────────────────────────────────────────────────────

class TestToUInt8:
    def test_zero_maps_to_128(self):
        t = ToUInt8()
        x = torch.tensor([0.0])
        assert t(x).item() == pytest.approx(128.0)

    def test_output_shape_unchanged(self):
        t = ToUInt8()
        x = torch.randn(2, 2, 4, 4)
        assert t(x).shape == x.shape

    def test_formula_correctness(self):
        # formula: 128 + (255/40) * x  → then round
        t = ToUInt8()
        x = torch.tensor([40.0])
        expected = round(128 + 255 / 40 * 40)
        assert t(x).item() == pytest.approx(expected)


# ── PILToTensor ───────────────────────────────────────────────────────────────

class TestPILToTensor:
    def test_output_shape_chw(self):
        t = PILToTensor()
        img = Image.fromarray(np.zeros((16, 12, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 16, 12)

    def test_output_is_tensor(self):
        t = PILToTensor()
        img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        assert isinstance(t(img), torch.Tensor)

    def test_values_preserved(self):
        t = PILToTensor()
        arr = np.full((4, 4, 3), 200, dtype=np.uint8)
        img = Image.fromarray(arr)
        out = t(img)
        assert int(out[0, 0, 0].item()) == 200


# ── ToFloat ───────────────────────────────────────────────────────────────────

class TestToFloat:
    def test_converts_to_float(self):
        t = ToFloat()
        x = torch.randint(0, 255, (3, 4, 4)).byte()
        out = t(x)
        assert out.dtype == torch.float32

    def test_values_unchanged(self):
        t = ToFloat()
        x = torch.tensor([1, 2, 3], dtype=torch.uint8)
        out = t(x)
        assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]))


# ── ResizeImproved ────────────────────────────────────────────────────────────

class TestResizeImproved:
    def test_resizes_smaller_edge_to_target(self):
        t = ResizeImproved(size=8)
        img = Image.fromarray(np.zeros((20, 10, 3), dtype=np.uint8))  # h=20, w=10
        out = t(img)
        # smaller edge is w=10; should become 8 → h scales to 20*(8/10)=16
        assert min(out.size) == 8

    def test_output_is_pil_image(self):
        t = ResizeImproved(size=16)
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        assert isinstance(t(img), Image.Image)

    def test_square_image_becomes_target_size(self):
        t = ResizeImproved(size=12)
        img = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8))
        out = t(img)
        assert out.size == (12, 12)


# ── ToCFHW_ToFloat ────────────────────────────────────────────────────────────

class TestToCFHW_ToFloat:
    def test_output_shape(self):
        # (F, H, W, C) → (C, F, H, W)
        t = ToCFHW_ToFloat()
        x = torch.randint(0, 255, (5, 8, 8, 3))
        out = t(x)
        assert out.shape == (3, 5, 8, 8)

    def test_output_dtype_float(self):
        t = ToCFHW_ToFloat()
        x = torch.randint(0, 255, (4, 6, 6, 2))
        assert t(x).dtype == torch.float32


# ── ToFCHW ────────────────────────────────────────────────────────────────────

class TestToFCHW:
    def test_output_shape(self):
        # (C, F, H, W) → (F, C, H, W)
        t = ToFCHW()
        x = torch.randn(3, 5, 8, 8)
        out = t(x)
        assert out.shape == (5, 3, 8, 8)

    def test_roundtrip_with_cfhw(self):
        """ToCFHW_ToFloat followed by ToFCHW should restore the frame dimension order."""
        x_fhwc = torch.randint(0, 255, (5, 8, 8, 3))
        cfhw = ToCFHW_ToFloat()(x_fhwc)
        fcmhw = ToFCHW()(cfhw)
        assert fcmhw.shape == (5, 3, 8, 8)
