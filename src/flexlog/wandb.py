"""Weights & Biases logging helpers for FlexLog."""

from speechbrain.utils.importutils import LazyModule
from speechbrain.utils.train_logger import WandBLogger
from numpy.typing import ArrayLike
from speechbrain import Stage
import torch

from flexlog.flex import _get_epoch


wandb = LazyModule(
    name="wandb",
    target="wandb",
    package=None
)


class FlexWandBLogger(WandBLogger):
    """SpeechBrain W&B logger with convenience methods for common artifacts."""

    def save_image(
        self,
        key: str,
        image: ArrayLike,
        stats_meta: dict | None = None
    ):
        """Log an image to the active W&B run.

        Arguments
        ---------
        key : str
            Name under which the image is stored in W&B.
        image : ArrayLike
            Image data accepted by ``wandb.Image``.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value is used as the W&B step.
        """
        self.log(
            values={key: wandb.Image(image)},
            stats_meta=stats_meta
        )

    def save_text(self, key: str, text: str, stats_meta: dict | None = None):
        """Log text to the active W&B run.

        Arguments
        ---------
        key : str
            Name under which the text is stored in W&B.
        text : str
            Text value to log.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value is used as the W&B step.
        """
        self.log(
            values={key: text},
            stats_meta=stats_meta
        )

    def log_audio(
        self,
        key: str,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Log audio to the active W&B run.

        Arguments
        ---------
        key : str
            Name under which the audio is stored in W&B.
        audio : torch.Tensor
            Waveform to log.
        sample_rate : int
            Sample rate of the waveform.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value is used as the W&B step.
        stage : Stage | None
            Optional SpeechBrain stage. Accepted for compatibility with other
            artifact loggers.
        """
        audio_data = audio.detach().cpu().numpy()
        self.log(
            values={key: wandb.Audio(audio_data, sample_rate=sample_rate)},
            stats_meta=stats_meta
        )

    def log(self, values: dict, stats_meta: dict | None = None):
        """Log values to W&B using epoch metadata as the step.

        If no W&B run is active, logging is skipped.

        Arguments
        ---------
        values : dict
            Mapping of metric or artifact names to values accepted by W&B.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value is passed to W&B as the logging step.
        """
        if self.run is None:
            return
        step = _get_epoch(stats_meta)
        self.run.log(
            values,
            step=step
        )
