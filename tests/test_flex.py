"""Tests for the aggregate FlexTrainLogger."""

import numpy as np
from speechbrain import Stage
from speechbrain.utils.train_logger import TrainLogger
import torch

from flexlog.flex import FlexTrainLogger


class StatsLogger(TrainLogger):
    """Test logger that records forwarded statistics."""

    def __init__(self):
        self.stats_calls = []

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=True,
    ):
        """Record a statistics logging call."""
        self.stats_calls.append(
            {
                "stats_meta": stats_meta,
                "train_stats": train_stats,
                "valid_stats": valid_stats,
                "test_stats": test_stats,
                "verbose": verbose,
            }
        )


class TextLogger(StatsLogger):
    """Test logger that records text artifact calls."""

    def __init__(self):
        super().__init__()
        self.text_calls = []

    def log_text(
        self,
        key,
        text,
        stats_meta=None,
        stage=None,
    ):
        """Record a text artifact logging call."""
        self.text_calls.append(
            {
                "key": key,
                "text": text,
                "stats_meta": stats_meta,
                "stage": stage,
            }
        )


class ImageLogger(StatsLogger):
    """Test logger that records image artifact calls."""

    def __init__(self):
        super().__init__()
        self.image_calls = []

    def log_image(
        self,
        key,
        image,
        stats_meta=None,
        stage=None,
    ):
        """Record an image artifact logging call."""
        self.image_calls.append(
            {
                "key": key,
                "image": image,
                "stats_meta": stats_meta,
                "stage": stage,
            }
        )


class AudioLogger(StatsLogger):
    """Test logger that records audio artifact calls."""

    def __init__(self):
        super().__init__()
        self.audio_calls = []

    def log_audio(
        self,
        key,
        audio,
        sample_rate=16000,
        stats_meta=None,
        stage=None,
    ):
        """Record an audio artifact logging call."""
        self.audio_calls.append(
            {
                "key": key,
                "audio": audio,
                "sample_rate": sample_rate,
                "stats_meta": stats_meta,
                "stage": stage,
            }
        )


class RichLogger(TextLogger, ImageLogger, AudioLogger):
    """Test logger that supports all artifact methods."""

    def __init__(self):
        StatsLogger.__init__(self)
        self.text_calls = []
        self.image_calls = []
        self.audio_calls = []


def test_enabled_outputs_only_returns_enabled_loggers():
    """Only enabled outputs are returned by enabled_outputs."""
    enabled = TextLogger()
    disabled = TextLogger()
    logger = FlexTrainLogger(
        outputs={"enabled": enabled, "disabled": disabled},
        enabled={"enabled": True, "disabled": False},
    )

    assert logger.enabled_outputs() == {"enabled": enabled}


def test_log_stats_forwards_only_to_enabled_loggers():
    """Stats are forwarded only to enabled loggers."""
    enabled = StatsLogger()
    disabled = StatsLogger()
    logger = FlexTrainLogger(
        outputs={"enabled": enabled, "disabled": disabled},
        enabled={"enabled": True, "disabled": False},
    )

    logger.log_stats(
        stats_meta={"epoch": 3},
        train_stats={"loss": 0.4},
        valid_stats={"loss": 0.5},
        test_stats={"loss": 0.6},
        verbose=False,
    )

    assert enabled.stats_calls == [
        {
            "stats_meta": {"epoch": 3},
            "train_stats": {"loss": 0.4},
            "valid_stats": {"loss": 0.5},
            "test_stats": {"loss": 0.6},
            "verbose": False,
        }
    ]
    assert disabled.stats_calls == []


def test_log_text_skips_disabled_and_unsupported_loggers():
    """Text logging skips disabled loggers and unsupported outputs."""
    enabled_text = TextLogger()
    disabled_text = TextLogger()
    stats_only = StatsLogger()
    logger = FlexTrainLogger(
        outputs={
            "enabled_text": enabled_text,
            "disabled_text": disabled_text,
            "stats_only": stats_only,
        },
        enabled={
            "enabled_text": True,
            "disabled_text": False,
            "stats_only": True,
        },
    )

    logger.log_text(
        key="caption",
        text="hello",
        stats_meta={"epoch": 2},
        stage=Stage.VALID,
    )

    assert enabled_text.text_calls == [
        {
            "key": "caption",
            "text": "hello",
            "stats_meta": {"epoch": 2},
            "stage": Stage.VALID,
        }
    ]
    assert disabled_text.text_calls == []
    assert not hasattr(stats_only, "text_calls")


def test_artifact_methods_only_call_loggers_supporting_that_method():
    """Artifact calls are dispatched only to loggers with matching methods."""
    rich = RichLogger()
    image_only = ImageLogger()
    audio_only = AudioLogger()
    text_only = TextLogger()
    logger = FlexTrainLogger(
        outputs={
            "rich": rich,
            "image_only": image_only,
            "audio_only": audio_only,
            "text_only": text_only,
        }
    )
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    audio = torch.ones(4)

    logger.log_image(
        key="plot",
        image=image,
        stats_meta={"epoch": 1},
        stage=Stage.TRAIN,
    )
    logger.log_audio(
        key="clip",
        audio=audio,
        sample_rate=8000,
        stats_meta={"epoch": 1},
        stage=Stage.TEST,
    )

    assert [call["key"] for call in rich.image_calls] == ["plot"]
    assert [call["key"] for call in image_only.image_calls] == ["plot"]
    assert text_only.text_calls == []

    assert [call["key"] for call in rich.audio_calls] == ["clip"]
    assert [call["key"] for call in audio_only.audio_calls] == ["clip"]
    assert not hasattr(image_only, "audio_calls")


def test_callable_outputs_are_initialized():
    """Callable output values are initialized before they are used."""
    logger = FlexTrainLogger(outputs={"text": TextLogger})

    output = logger.outputs["text"]
    assert isinstance(output, TextLogger)

    logger.log_text("message", "created")

    assert output.text_calls == [
        {
            "key": "message",
            "text": "created",
            "stats_meta": None,
            "stage": None,
        }
    ]
