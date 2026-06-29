"""Unit tests for configs/wandb_utils.py — wandb.init is mocked throughout."""

from unittest.mock import MagicMock, patch

import pytest

from configs.wandb_utils import init_wandb


class TestInitWandB:
    def test_calls_wandb_init(self):
        with patch("configs.wandb_utils.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            init_wandb("mstcn", "ImPerfectPour", "1", "BRP")
            mock_wandb.init.assert_called_once()

    def test_run_name_format(self):
        with patch("configs.wandb_utils.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            init_wandb("mstcn", "ImPerfectPour", "1", "BRP")
            _, kwargs = mock_wandb.init.call_args
            assert kwargs["name"] == "mstcn_ImPerfectPour_split1_BRP"

    def test_project_is_inverse_tas(self):
        with patch("configs.wandb_utils.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            init_wandb("mstcn", "JIGSAWS", "2", "I3D")
            _, kwargs = mock_wandb.init.call_args
            assert kwargs["project"] == "inverse_tas"

    def test_base_config_keys(self):
        with patch("configs.wandb_utils.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            init_wandb("ASFormer", "50Salads", "3", "BRP")
            _, kwargs = mock_wandb.init.call_args
            config = kwargs["config"]
            assert config["model"] == "ASFormer"
            assert config["dataset"] == "50Salads"
            assert config["split"] == "3"
            assert config["f_type"] == "BRP"

    def test_extra_config_merged(self):
        with patch("configs.wandb_utils.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            init_wandb(
                "mstcn", "IRIS", "1", "BRP",
                extra_config={"num_epochs": 50, "lr": 5e-4},
            )
            _, kwargs = mock_wandb.init.call_args
            config = kwargs["config"]
            assert config["num_epochs"] == 50
            assert config["lr"] == pytest.approx(5e-4)
            assert config["model"] == "mstcn"  # base config still present

    def test_returns_wandb_run(self):
        mock_run = MagicMock()
        with patch("configs.wandb_utils.wandb") as mock_wandb:
            mock_wandb.init.return_value = mock_run
            result = init_wandb("onlinetas", "GTEA", "1", "BRP")
            assert result is mock_run
