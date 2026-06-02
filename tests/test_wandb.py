"""Tests for the W&B FlexLog helper using mocks."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from speechbrain import Stage
import torch

import flexlog.wandb as wandb_module
from flexlog.wandb import FlexWandBLogger


def make_logger(run=None) -> FlexWandBLogger:
    """Create a W&B logger instance with a mocked run."""
    logger = FlexWandBLogger.__new__(FlexWandBLogger)
    logger.run = Mock() if run is None else run
    return logger


def test_log_uses_epoch_as_step():
    """The epoch metadata is passed to W&B as the step."""
    logger = make_logger()

    logger.log({"loss": 0.25}, stats_meta={"epoch": 7})
    assert logger.run is not None
    logger.run.log.assert_called_once_with({"loss": 0.25}, step=7)


def test_log_uses_none_step_without_epoch():
    """Logging without an epoch uses a None step."""
    logger = make_logger()

    logger.log({"loss": 0.25})
    assert logger.run is not None
    logger.run.log.assert_called_once_with({"loss": 0.25}, step=None)


def test_log_is_noop_without_active_run():
    """Logging is skipped when no W&B run is active."""
    logger = make_logger(run=None)
    logger.run = None

    logger.log({"loss": 0.25}, stats_meta={"epoch": 7})


def test_save_image_wraps_image_with_wandb_image(monkeypatch):
    """Images are wrapped with wandb.Image before logging."""
    image_artifact = object()
    fake_wandb = SimpleNamespace(Image=Mock(return_value=image_artifact))
    monkeypatch.setattr(wandb_module, "wandb", fake_wandb)
    logger = make_logger()
    image = np.zeros((4, 6, 3), dtype=np.uint8)

    logger.save_image("example", image, stats_meta={"epoch": 3})

    fake_wandb.Image.assert_called_once_with(image)
    assert logger.run is not None
    logger.run.log.assert_called_once_with(
        {"example": image_artifact},
        step=3,
    )


def test_save_text_logs_plain_text_value():
    """Text values are logged without additional wrapping."""
    logger = make_logger()

    logger.save_text("caption", "hello", stats_meta={"epoch": 2})
    assert logger.run is not None

    logger.run.log.assert_called_once_with({"caption": "hello"}, step=2)


def test_log_audio_wraps_tensor_with_wandb_audio(monkeypatch):
    """Audio tensors are converted to NumPy and wrapped with wandb.Audio."""
    audio_artifact = object()
    fake_wandb = SimpleNamespace(Audio=Mock(return_value=audio_artifact))
    monkeypatch.setattr(wandb_module, "wandb", fake_wandb)
    logger = make_logger()
    audio = torch.arange(5, dtype=torch.float32)

    logger.log_audio(
        "sample",
        audio,
        sample_rate=8000,
        stats_meta={"epoch": 5},
        stage=Stage.TEST,
    )

    audio_data = fake_wandb.Audio.call_args.args[0]
    assert np.array_equal(audio_data, audio.numpy())
    assert fake_wandb.Audio.call_args.kwargs == {"sample_rate": 8000}
    assert logger.run is not None

    logger.run.log.assert_called_once_with(
        {"sample": audio_artifact},
        step=5,
    )
