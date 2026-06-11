"""An aggregate logger with multple loggers

Authors
* Artem Ploujnikov 2026
"""

from speechbrain import Stage
import torch

from speechbrain.utils.train_logger import TrainLogger, main_process_only
from typing import Any, Protocol, runtime_checkable
from numpy.typing import ArrayLike


EPOCH_KEYS = ("epoch", "Epoch", "Epoch loaded")


def _get_epoch(stats_meta: dict | None = None):
    """Return the epoch value from SpeechBrain metadata.

    SpeechBrain recipes use a few different epoch metadata keys. If multiple
    keys are present, the first key in ``EPOCH_KEYS`` wins.

    Arguments
    ---------
    stats_meta : dict | None
        Metadata dictionary that may contain an epoch value.

    Returns
    -------
    object | None
        Epoch value if present, otherwise ``None``.
    """
    if stats_meta is None:
        return None
    for key in EPOCH_KEYS:
        if key in stats_meta:
            return stats_meta[key]
    return None


@runtime_checkable
class SupportsImageLog(Protocol):
    """A protocol for image loggers"""
    def log_image(
        self,
        key: str,
        image: ArrayLike,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Log an image artifact.

        Arguments
        ---------
        key : str
            Name under which the image is logged.
        image : ArrayLike
            Image data to log.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value can be used by downstream loggers.
        stage : Stage | None
            Optional SpeechBrain stage associated with the logged image.
        """
        ...


@runtime_checkable
class SupportsTextLog(Protocol):
    """A protocol for text loggers"""
    def log_text(
        self,
        key: str,
        text: str,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Log a text artifact.

        Arguments
        ---------
        key : str
            Name under which the text is logged.
        text : str
            Text value to log.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value can be used by downstream loggers.
        stage : Stage | None
            Optional SpeechBrain stage associated with the logged text.
        """


@runtime_checkable
class SupportsAudioLog(Protocol):
    """A protocol for audio loggers"""
    def log_audio(
        self,
        key: str,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Log an audio artifact.

        Arguments
        ---------
        key : str
            Name under which the audio is logged.
        audio : torch.Tensor
            The waveform to log
        sample_rate : int
            The sample rate
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value can be used by downstream loggers.
        stage : Stage | None
            Optional SpeechBrain stage associated with the logged audio.
        """


class FlexTrainLogger(TrainLogger):
    """A flexible train logger that can aggregate multiple loggers
    together and enable/disable them on request

    Arguments
    ---------
    outputs : dict
        Mapping of output names to logger instances or callables that create
        logger instances.
    enabled : dict | None
        Optional mapping of output names to booleans. Outputs omitted from the
        mapping are treated as disabled.
    runtime_toggle : bool
        enables runtime toggling of the outputs

    """
    def __init__(
            self,
            outputs: dict, enabled: dict | None = None,
            runtime_toggle: bool = False):
        if enabled is None:
            enabled = {key: True for key in self.outputs.keys()}
        self.enabled = enabled
        self.runtime_toggle = runtime_toggle
        self.outputs = {
            key: self._init_output(output)
            for key, output in outputs.items()
            if runtime_toggle or self.enabled.get(key)
        }

    def _init_output(self, output: Any) -> TrainLogger:
        """Return a logger instance for an output configuration.

        If ``output`` is callable and is not already a ``TrainLogger``, it is
        called with no arguments and the result is used as the logger.

        Arguments
        ---------
        output : Any
            Logger instance or zero-argument factory.

        Returns
        -------
        TrainLogger
            Initialized logger output.
        """
        if not isinstance(output, TrainLogger) and callable(output):
            output = output()
        return output

    def enabled_outputs(self):
        """Returns only enabled outputs

        Returns
        -------
        result : dict
            Enabled outputs"""
        return {
            key: output
            for key, output in self.outputs.items()
            if self.enabled.get(key)
        }

    @main_process_only
    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=True,
    ):
        """Log training statistics to every enabled output.

        Arguments
        ---------
        stats_meta : dict
            Metadata associated with the stats, such as epoch or learning
            rate.
        train_stats : dict | None
            Training statistics to forward.
        valid_stats : dict | None
            Validation statistics to forward.
        test_stats : dict | None
            Test statistics to forward.
        verbose : bool
            Whether downstream loggers should also emit verbose output.
        """
        for logger in self.enabled_outputs().values():
            logger.log_stats(
                stats_meta,
                train_stats=train_stats,
                valid_stats=valid_stats,
                test_stats=test_stats,
                verbose=verbose
            )

    def log_image(
        self,
        key: str,
        image: ArrayLike,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Log an image artifact to all compatible outputs.

        Arguments
        ---------
        key : str
            Name under which the image is logged.
        image : ArrayLike
            Image data to log.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value can be used by downstream loggers.
        stage : Stage | None
            Optional SpeechBrain stage associated with the logged image.
        """
        for logger in self.enabled_outputs().values():
            if isinstance(logger, SupportsImageLog):
                logger.log_image(
                    key=key,
                    image=image,
                    stats_meta=stats_meta,
                    stage=stage
                )

    def log_text(
        self,
        key: str,
        text: str,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Log a text artifact to all compatible outputs.

        Arguments
        ---------
        key : str
            Name under which the text is logged.
        text : str
            Text value to log.
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value can be used by downstream loggers.
        stage : Stage | None
            Optional SpeechBrain stage associated with the logged text.
        """
        for logger in self.enabled_outputs().values():
            if isinstance(logger, SupportsTextLog):
                logger.log_text(
                    key=key,
                    text=text,
                    stats_meta=stats_meta,
                    stage=stage
                )

    def log_audio(
        self,
        key: str,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        stats_meta: dict | None = None,
        stage: Stage | None = None
    ):
        """Log an audio artifact to all compatible outputs.

        Arguments
        ---------
        key : str
            Name under which the audio is logged.
        audio : torch.Tensor
            The waveform to log
        sample_rate : int
            The sample rate
        stats_meta : dict | None
            Optional metadata dictionary. If it contains an epoch key, that
            value can be used by downstream loggers.
        stage : Stage | None
            Optional SpeechBrain stage associated with the logged audio.
        """
        for logger in self.enabled_outputs().values():
            if isinstance(logger, SupportsAudioLog):
                logger.log_audio(
                    key=key,
                    audio=audio,
                    sample_rate=sample_rate,
                    stats_meta=stats_meta,
                    stage=stage
                )
