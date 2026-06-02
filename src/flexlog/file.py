from os import PathLike
from pathlib import Path

from speechbrain import Stage
from speechbrain.utils.train_logger import FileTrainLogger
from speechbrain.dataio import audio_io
from numpy.typing import ArrayLike
from PIL import Image
import numpy as np
import torch


class FlexFileTrainLogger(FileTrainLogger):
    def __init__(
            self,
            save_file: str | PathLike,
            precision: int = 2,
            progress_folder: str | PathLike | None = None
    ):
        """Text logger of training information.

        Arguments
        ---------
        save_file : str | PathLike
            The file to use for logging train information.
        precision : int
            Number of decimal places to display. Default 2, example: 1.35e-5.
        progress_folder : str | PathLike | None
            the folder where artifacts will be saved
        """
        super().__init__(save_file=save_file,
                         precision=precision)
        if progress_folder is None:
            progress_folder = Path(save_file).parent / "progress"

        self.progress_folder = Path(progress_folder)
        self.progress_folder.mkdir(
            exist_ok=True,
            parents=True
        )

    """Text logger of training information.

    Arguments
    ---------
    save_file : str
        The file to use for logging train information.
    precision : int
        Number of decimal places to display. Default 2, example: 1.35e-5.
    """
    def log_image(
        self,
        key: str,
        image: ArrayLike,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Save an image artifact to the progress folder.

        Arguments
        ---------
        key : str
            Base filename for the saved image.
        image : ArrayLike
            Image data to save.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains ``"epoch"``, that
            value is added to the artifact path.
        stage : Stage | None
            Optional SpeechBrain stage. When provided, artifacts are saved
            under a lowercase stage folder before the epoch folder.
        """
        progress_folder = self.get_progress_folder(
            stats_meta=stats_meta,
            stage=stage
        )
        file_name = progress_folder / f"{key}.png"

        with Image.fromarray(np.array(image)) as pil_image:
            pil_image.save(file_name)

    def log_text(
        self,
        key: str,
        text: str,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Save a text artifact to the progress folder.

        Arguments
        ---------
        key : str
            Base filename for the saved text.
        text : str
            Text value to log.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains ``"epoch"``, that
            value is added to the artifact path.
        stage : Stage | None
            Optional SpeechBrain stage. When provided, artifacts are saved
            under a lowercase stage folder before the epoch folder.
        """
        progress_folder = self.get_progress_folder(
            stats_meta=stats_meta,
            stage=stage
        )
        file_name = progress_folder / f"{key}.txt"
        with open(file_name, "w") as text_file:
            print(text, file=text_file)

    def log_audio(
        self,
        key: str,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Save an audio artifact to the progress folder.

        Arguments
        ---------
        key : str
            Base filename for the saved audio.
        audio : torch.Tensor
            The waveform to log
        stats_meta : dict | None
            Optional metadata dictionary. If it contains ``"epoch"``, that
            value is added to the artifact path.
        stage : Stage | None
            Optional SpeechBrain stage. When provided, artifacts are saved
            under a lowercase stage folder before the epoch folder.
        """
        progress_folder = self.get_progress_folder(
            stats_meta=stats_meta,
            stage=stage
        )
        file_name = progress_folder / f"{key}.wav"
        audio_io.save(file_name, audio, sample_rate=sample_rate)

    def get_progress_folder(
        self,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        epoch = None
        if stats_meta is not None:
            epoch = stats_meta.get("epoch")
        progress_folder = self.progress_folder
        if stage is not None:
            progress_folder = progress_folder / stage.name.lower()
        if epoch is not None:
            progress_folder = progress_folder / str(epoch)
        progress_folder.mkdir(exist_ok=True, parents=True)

        return progress_folder
