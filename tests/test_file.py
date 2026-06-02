"""Tests for file-based FlexLog artifact logging."""

from pathlib import Path
import wave

import numpy as np
from PIL import Image
from speechbrain import Stage
import torch

from flexlog.file import FlexFileTrainLogger


def make_logger(tmp_path: Path) -> FlexFileTrainLogger:
    """Create a file logger rooted in a pytest temporary directory."""
    return FlexFileTrainLogger(save_file=tmp_path / "train.log")


def test_default_progress_folder_is_next_to_save_file(tmp_path):
    """The default progress folder is created next to the train log."""
    logger = make_logger(tmp_path)

    assert logger.progress_folder == tmp_path / "progress"
    assert logger.progress_folder.is_dir()


def test_log_text_writes_to_stage_and_epoch_folder(tmp_path):
    """Text artifacts are written below stage and epoch folders."""
    logger = make_logger(tmp_path)

    logger.log_text(
        key="transcript",
        text="hello world",
        stats_meta={"epoch": 4},
        stage=Stage.VALID,
    )

    output_path = tmp_path / "progress" / "valid" / "4" / "transcript.txt"
    assert output_path.is_file()
    assert output_path.read_text() == "hello world\n"


def test_log_image_writes_png_to_progress_folder(tmp_path):
    """Image artifacts are written as readable PNG files."""
    logger = make_logger(tmp_path)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 1] = 255

    logger.log_image(key="sample", image=image)

    output_path = tmp_path / "progress" / "sample.png"
    assert output_path.is_file()

    with Image.open(output_path) as output_image:
        assert output_image.size == (12, 8)
        assert output_image.mode == "RGB"


def test_log_audio_writes_wav_to_stage_and_epoch_folder(tmp_path):
    """Audio artifacts are written as WAV files below stage and epoch folders."""
    logger = make_logger(tmp_path)

    logger.log_audio(
        key="tone",
        audio=torch.zeros(160),
        sample_rate=16000,
        stats_meta={"epoch": 2},
        stage=Stage.TEST,
    )

    output_path = tmp_path / "progress" / "test" / "2" / "tone.wav"
    assert output_path.is_file()

    with wave.open(str(output_path)) as output_audio:
        assert output_audio.getframerate() == 16000
        assert output_audio.getnframes() == 160
