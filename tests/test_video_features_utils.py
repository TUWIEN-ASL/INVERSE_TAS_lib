"""Unit tests for models/video_features/utils/utils.py and
models/video_features/models/_base/base_extractor.py.

Covers pure utility functions (make_path, form_slices, numpy I/O,
dp_state_to_normal) and the BaseExtractor feature-saving pipeline
(action_on_extraction, is_already_exist).  No real video or GPU needed.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_BASE = os.path.join(os.path.dirname(__file__), "..")


def _load(unique_name, relpath):
    path = os.path.join(_BASE, relpath)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load utils; also register under its canonical import path so that
# base_extractor.py's `from models.video_features.utils.utils import ...`
# finds the real module even if test_predict.py registered a parent mock.
_vf_utils = _load("vf_utils", "models/video_features/utils/utils.py")
sys.modules.setdefault("models.video_features.utils", type(sys)("models.video_features.utils"))
sys.modules["models.video_features.utils.utils"] = _vf_utils

make_path        = _vf_utils.make_path
form_slices      = _vf_utils.form_slices
load_numpy       = _vf_utils.load_numpy
write_numpy      = _vf_utils.write_numpy
dp_state_to_normal = _vf_utils.dp_state_to_normal

_vf_base = _load("vf_base_extractor", "models/video_features/models/_base/base_extractor.py")
BaseExtractor = _vf_base.BaseExtractor


# ── Concrete extractor for testing abstract BaseExtractor ─────────────────────

class _TestExtractor(BaseExtractor):
    """Minimal concrete subclass; output_feat_keys must be set before use."""

    def __init__(self, on_extraction, output_path, feat_keys=("rgb",)):
        super().__init__(
            feature_type="test",
            on_extraction=on_extraction,
            tmp_path=str(output_path),
            output_path=str(output_path),
            keep_tmp_files=False,
            device="cpu",
        )
        self.output_feat_keys = list(feat_keys)

    def extract(self, video_path):
        return {}


# ── make_path ─────────────────────────────────────────────────────────────────

class TestMakePath:
    def test_uses_video_stem(self):
        p = make_path("/out", "/videos/test_video.mp4", "rgb", ".npy")
        assert "test_video" in p

    def test_appends_key_and_extension(self):
        p = make_path("/out", "/videos/clip.mp4", "flow", ".npy")
        assert p.endswith("flow.npy")

    def test_output_in_correct_directory(self):
        p = make_path("/features", "/videos/v.mp4", "rgb", ".npy")
        assert p.startswith("/features")

    def test_no_original_directory_in_filename(self):
        """The video's parent directory must not appear in the output filename."""
        p = make_path("/out", "/some/deep/path/video.mp4", "rgb", ".npy")
        fname = Path(p).name
        assert "path" not in fname
        assert "some" not in fname

    def test_pickle_extension(self):
        p = make_path("/out", "vid.mp4", "rgb", ".pkl")
        assert p.endswith(".pkl")


# ── form_slices ───────────────────────────────────────────────────────────────

class TestFormSlices:
    def test_correct_number_of_slices(self):
        # (100 - 15) // 15 + 1 = 6
        slices = form_slices(100, stack_size=15, step_size=15)
        assert len(slices) == 6

    def test_first_slice_starts_at_zero(self):
        slices = form_slices(100, 10, 10)
        assert slices[0] == (0, 10)

    def test_each_slice_has_correct_width(self):
        slices = form_slices(100, 20, 10)
        for start, end in slices:
            assert end - start == 20

    def test_no_slices_when_size_lt_stack(self):
        slices = form_slices(5, stack_size=10, step_size=5)
        assert len(slices) == 0

    def test_exactly_one_slice_when_size_equals_stack(self):
        slices = form_slices(10, stack_size=10, step_size=5)
        assert len(slices) == 1
        assert slices[0] == (0, 10)

    def test_non_overlapping_slices_cover_range(self):
        slices = form_slices(50, stack_size=10, step_size=10)
        assert slices[-1][1] <= 50


# ── numpy I/O roundtrip ───────────────────────────────────────────────────────

class TestNumpyIO:
    def test_write_and_load_float_array(self, tmp_path):
        arr = np.random.randn(10, 512).astype(np.float32)
        fpath = str(tmp_path / "features.npy")
        write_numpy(fpath, arr)
        loaded = load_numpy(fpath)
        np.testing.assert_array_almost_equal(arr, loaded)

    def test_write_and_load_int_array(self, tmp_path):
        arr = np.array([0, 1, 2, 3], dtype=np.int64)
        fpath = str(tmp_path / "labels.npy")
        write_numpy(fpath, arr)
        loaded = load_numpy(fpath)
        np.testing.assert_array_equal(arr, loaded)

    def test_shape_preserved(self, tmp_path):
        arr = np.zeros((5, 3, 224, 224), dtype=np.float32)
        fpath = str(tmp_path / "frames.npy")
        write_numpy(fpath, arr)
        loaded = load_numpy(fpath)
        assert loaded.shape == arr.shape


# ── dp_state_to_normal ────────────────────────────────────────────────────────

class TestDpStateToNormal:
    def test_removes_module_prefix(self):
        sd = {"module.layer.weight": torch.ones(3), "module.layer.bias": torch.zeros(3)}
        out = dp_state_to_normal(sd)
        assert "layer.weight" in out
        assert "layer.bias" in out

    def test_no_module_prefix_in_output(self):
        sd = {"module.conv.weight": torch.ones(1)}
        out = dp_state_to_normal(sd)
        assert not any(k.startswith("module") for k in out)

    def test_values_preserved(self):
        w = torch.tensor([1.0, 2.0])
        sd = {"module.w": w}
        out = dp_state_to_normal(sd)
        assert torch.allclose(out["w"], w)

    def test_multiple_keys(self):
        sd = {f"module.layer{i}.weight": torch.ones(1) for i in range(5)}
        out = dp_state_to_normal(sd)
        assert len(out) == 5


# ── BaseExtractor.action_on_extraction ───────────────────────────────────────

class TestActionOnExtraction:
    def test_save_numpy_creates_file(self, tmp_path):
        ext = _TestExtractor("save_numpy", tmp_path, feat_keys=("rgb",))
        feats = {"rgb": np.random.randn(10, 2048).astype(np.float32)}
        ext.action_on_extraction(feats, "video.mp4")
        saved = list(tmp_path.glob("*.npy"))
        assert len(saved) == 1

    def test_saved_array_matches_input(self, tmp_path):
        ext = _TestExtractor("save_numpy", tmp_path, feat_keys=("rgb",))
        arr = np.arange(30, dtype=np.float32).reshape(3, 10)
        ext.action_on_extraction({"rgb": arr}, "vid.mp4")
        npy_path = next(tmp_path.glob("*.npy"))
        loaded = np.load(str(npy_path))
        np.testing.assert_array_equal(arr, loaded)

    def test_print_mode_does_not_create_files(self, tmp_path, capsys):
        ext = _TestExtractor("print", tmp_path, feat_keys=("rgb",))
        ext.action_on_extraction({"rgb": np.ones((5, 8))}, "vid.mp4")
        assert list(tmp_path.glob("*")) == []

    def test_multiple_keys_create_multiple_files(self, tmp_path):
        ext = _TestExtractor("save_numpy", tmp_path, feat_keys=("rgb", "flow"))
        feats = {
            "rgb":  np.random.randn(10, 2048).astype(np.float32),
            "flow": np.random.randn(10, 2048).astype(np.float32),
        }
        ext.action_on_extraction(feats, "video.mp4")
        saved = list(tmp_path.glob("*.npy"))
        assert len(saved) == 2


# ── BaseExtractor.is_already_exist ────────────────────────────────────────────

class TestIsAlreadyExist:
    def test_print_mode_always_returns_false(self, tmp_path):
        ext = _TestExtractor("print", tmp_path)
        assert ext.is_already_exist("any_video.mp4") is False

    def test_returns_false_when_file_missing(self, tmp_path):
        ext = _TestExtractor("save_numpy", tmp_path, feat_keys=("rgb",))
        assert ext.is_already_exist("nonexistent.mp4") is False

    def test_returns_true_when_all_files_exist(self, tmp_path):
        ext = _TestExtractor("save_numpy", tmp_path, feat_keys=("rgb",))
        # Write the expected file manually
        fpath = make_path(str(tmp_path), "video.mp4", "rgb", ".npy")
        write_numpy(fpath, np.ones(5))
        assert ext.is_already_exist("video.mp4") is True
