"""Unit tests for models/BridgePrompt/clip/model.py — CLIP building blocks and
the ImageCLIP encoder wrapper used during feature extraction.

All tests run on CPU without requiring pre-trained weights or CLIP checkpoints.
Small architectural configs are used throughout to keep the tests fast.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

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


_clip = _load("brp_clip_model", "models/BridgePrompt/clip/model.py")

QuickGELU              = _clip.QuickGELU
LayerNorm              = _clip.LayerNorm
ResidualAttentionBlock = _clip.ResidualAttentionBlock
Transformer            = _clip.Transformer
VisualTransformer      = _clip.VisualTransformer

# ImageCLIP lives in extract_frame_features.py, which has heavy script-level
# imports (datasets, YAML configs).  Redefine it inline — it is only 7 lines.
class ImageCLIP(nn.Module):
    """Thin CLIP image-encoder wrapper (mirrors BridgePrompt/extract_frame_features.py)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, image):
        return self.model.encode_image(image)


# ── Small architecture constants ──────────────────────────────────────────────

D_MODEL = 16     # embedding / feature dimension
N_HEAD  = 4      # attention heads (D_MODEL must be divisible by N_HEAD)
SEQ_LEN = 10     # sequence length (L)
BATCH   = 2      # batch size
# VisualTransformer settings
VT_RES   = 8    # input image resolution
VT_PATCH = 4    # patch size → num_patches = (8/4)^2 = 4
VT_OUT   = 12   # output embedding dimension


# ── QuickGELU ─────────────────────────────────────────────────────────────────

class TestQuickGELU:
    def test_output_shape_preserved(self):
        m = QuickGELU()
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        assert m(x).shape == x.shape

    def test_output_differs_from_input(self):
        m = QuickGELU()
        x = torch.randn(4, 4)
        assert not torch.allclose(m(x), x)

    def test_zero_maps_to_zero(self):
        m = QuickGELU()
        x = torch.zeros(3)
        assert torch.allclose(m(x), torch.zeros(3))

    def test_output_dtype_preserved(self):
        m = QuickGELU()
        x = torch.randn(2, 2)
        assert m(x).dtype == x.dtype


# ── LayerNorm ─────────────────────────────────────────────────────────────────

class TestLayerNorm:
    def test_output_shape_preserved(self):
        m = LayerNorm(D_MODEL)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        assert m(x).shape == x.shape

    def test_handles_fp32_input(self):
        m = LayerNorm(D_MODEL)
        x = torch.randn(2, D_MODEL, dtype=torch.float32)
        out = m(x)
        assert out.dtype == torch.float32

    def test_handles_fp16_input(self):
        """LayerNorm upcasts fp16 → fp32 internally, then returns fp16."""
        m = LayerNorm(D_MODEL)
        x = torch.randn(2, D_MODEL, dtype=torch.float16)
        out = m(x)
        assert out.dtype == torch.float16

    def test_normalizes_mean_close_to_zero(self):
        m = LayerNorm(D_MODEL)
        m.eval()
        x = torch.randn(1, D_MODEL) * 100
        out = m(x)
        assert out.mean().abs().item() < 0.1


# ── ResidualAttentionBlock ────────────────────────────────────────────────────

class TestResidualAttentionBlock:
    @pytest.fixture
    def block(self):
        return ResidualAttentionBlock(d_model=D_MODEL, n_head=N_HEAD)

    def test_output_shape(self, block):
        # Input to MultiheadAttention is (L, N, D) — sequence-first
        x = torch.randn(SEQ_LEN, BATCH, D_MODEL)
        assert block(x).shape == (SEQ_LEN, BATCH, D_MODEL)

    def test_output_dtype_float32(self, block):
        x = torch.randn(SEQ_LEN, BATCH, D_MODEL)
        assert block(x).dtype == torch.float32

    def test_gradients_flow(self, block):
        x = torch.randn(SEQ_LEN, BATCH, D_MODEL, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None

    def test_eval_mode_deterministic(self, block):
        block.eval()
        x = torch.randn(SEQ_LEN, BATCH, D_MODEL)
        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)
        assert torch.allclose(out1, out2)


# ── Transformer ───────────────────────────────────────────────────────────────

class TestTransformer:
    def test_output_shape(self):
        m = Transformer(width=D_MODEL, layers=2, heads=N_HEAD)
        x = torch.randn(SEQ_LEN, BATCH, D_MODEL)
        assert m(x).shape == (SEQ_LEN, BATCH, D_MODEL)

    def test_single_layer(self):
        m = Transformer(width=D_MODEL, layers=1, heads=N_HEAD)
        x = torch.randn(SEQ_LEN, BATCH, D_MODEL)
        assert m(x).shape == (SEQ_LEN, BATCH, D_MODEL)

    def test_gradients_flow(self):
        m = Transformer(width=D_MODEL, layers=2, heads=N_HEAD)
        x = torch.randn(SEQ_LEN, BATCH, D_MODEL, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


# ── VisualTransformer ─────────────────────────────────────────────────────────

class TestVisualTransformer:
    @pytest.fixture
    def model(self):
        return VisualTransformer(
            input_resolution=VT_RES,
            patch_size=VT_PATCH,
            width=D_MODEL,
            layers=1,
            heads=N_HEAD,
            output_dim=VT_OUT,
            joint=False,
            if_proj=True,
        )

    def test_output_shape(self, model):
        x = torch.randn(BATCH, 3, VT_RES, VT_RES)
        out = model(x)
        assert out.shape == (BATCH, VT_OUT)

    def test_output_dtype_float32(self, model):
        x = torch.randn(BATCH, 3, VT_RES, VT_RES)
        assert model(x).dtype == torch.float32

    def test_no_proj_returns_width_dim(self):
        """With if_proj=False, output dim equals the transformer width."""
        m = VisualTransformer(
            input_resolution=VT_RES, patch_size=VT_PATCH,
            width=D_MODEL, layers=1, heads=N_HEAD,
            output_dim=VT_OUT, joint=False, if_proj=False,
        )
        x = torch.randn(BATCH, 3, VT_RES, VT_RES)
        assert m(x).shape == (BATCH, D_MODEL)

    def test_gradients_flow(self, model):
        x = torch.randn(BATCH, 3, VT_RES, VT_RES)
        model(x).sum().backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_single_image_input(self, model):
        x = torch.randn(1, 3, VT_RES, VT_RES)
        assert model(x).shape == (1, VT_OUT)


# ── ImageCLIP ─────────────────────────────────────────────────────────────────

class TestImageCLIP:
    def test_delegates_to_encode_image(self):
        """ImageCLIP.forward must call model.encode_image exactly once."""
        mock_clip = MagicMock()
        expected = torch.randn(BATCH, VT_OUT)
        mock_clip.encode_image.return_value = expected

        wrapper = ImageCLIP(mock_clip)
        x = torch.randn(BATCH, 3, VT_RES, VT_RES)
        result = wrapper(x)

        mock_clip.encode_image.assert_called_once_with(x)
        assert torch.allclose(result, expected)

    def test_output_is_encode_image_result(self):
        """Output of ImageCLIP must equal the underlying encoder output."""
        visual = VisualTransformer(
            input_resolution=VT_RES, patch_size=VT_PATCH,
            width=D_MODEL, layers=1, heads=N_HEAD,
            output_dim=VT_OUT, joint=False, if_proj=True,
        )

        class _MinimalCLIP:
            def encode_image(self, img):
                return visual(img)

        wrapper = ImageCLIP(_MinimalCLIP())
        x = torch.randn(BATCH, 3, VT_RES, VT_RES)
        out = wrapper(x)
        assert out.shape == (BATCH, VT_OUT)

    def test_wrapper_is_nn_module(self):
        wrapper = ImageCLIP(MagicMock())
        assert isinstance(wrapper, nn.Module)
