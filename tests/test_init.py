"""Tests for the public FlexLog package API."""

import flexlog
from flexlog.file import FlexFileTrainLogger
from flexlog.flex import (
    FlexTrainLogger,
    SupportsAudioLog,
    SupportsImageLog,
    SupportsTextLog,
)
from flexlog.wandb import FlexWandBLogger


def test_public_api_reexports_common_logger_classes():
    """Common logger classes are importable from the package root."""
    assert flexlog.FlexTrainLogger is FlexTrainLogger
    assert flexlog.FlexFileTrainLogger is FlexFileTrainLogger
    assert flexlog.FlexWandBLogger is FlexWandBLogger


def test_public_api_reexports_protocols():
    """Artifact logger protocols are importable from the package root."""
    assert flexlog.SupportsAudioLog is SupportsAudioLog
    assert flexlog.SupportsImageLog is SupportsImageLog
    assert flexlog.SupportsTextLog is SupportsTextLog
