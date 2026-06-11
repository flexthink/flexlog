# FlexLog

FlexLog adds lightweight artifact logging helpers around SpeechBrain train
loggers.

It provides:

- `FlexTrainLogger`: aggregates multiple loggers and can enable or disable
  each output by name.
- `FlexFileTrainLogger`: writes image, text, and audio artifacts to a local
  progress folder.
- `FlexWandBLogger`: logs artifacts to Weights & Biases.

## Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

For tests:

```bash
pip install -e ".[test]"
```

If you want to use `FlexWandBLogger`, make sure W&B is available:

```bash
pip install "wandb>=0.16"
```

## Basic Usage

### HyperPyYAML

The intended use is to instantiate `FlexTrainLogger` from HyperPyYAML and pass
it into your SpeechBrain training code as the train logger.

```yaml
output_folder: results/example
logging_file_enabled: True
logging_wandb_enabled: False
logging_wandb_project: my_project
logging_wandb_name: my_run

train_logger: !new:flexlog.flex.FlexTrainLogger
    outputs:
        file: !name:flexlog.file.FlexFileTrainLogger
            save_file: !ref <output_folder>/train_log.txt
            progress_folder: !ref <output_folder>/progress

        wandb: !name:flexlog.wandb.FlexWandBLogger
            initializer: !name:wandb.init
                project: !ref <logging_wandb_project>
                name: !ref <logging_wandb_name>
                dir: !ref <output_folder>/wandb
                reinit: True
                resume: allow

    enabled:
        file: !ref <logging_file_enabled>
        wandb: !ref <logging_wandb_enabled>
    runtime_toggle: False
```

Load it the same way as other SpeechBrain HyperPyYAML configuration:

```python
from hyperpyyaml import load_hyperpyyaml

with open("hparams.yaml", encoding="utf-8") as fin:
    hparams = load_hyperpyyaml(fin)

train_logger = hparams["train_logger"]
```

The keys under `outputs` are logger names. The `enabled` mapping controls which
outputs receive logs, so you can keep W&B configured but disabled for local
runs. The enable flags and W&B run identifiers are top-level values so
SpeechBrain command-line overrides can change them directly.

`runtime_toggle` controls when outputs are initialized:

- With `runtime_toggle: False` (the default), only outputs enabled when
  `FlexTrainLogger` is created are initialized. This is useful when an output
  requires an external provider, installed integration, account, or
  credentials that are not available for every run.
- With `runtime_toggle: True`, all outputs are initialized, including disabled
  ones. You can then change the `enabled` mapping while the program is running
  to start or stop forwarding logs without recreating the logger.

To defer initialization in HyperPyYAML, define each output with `!name:` as
shown above. Using `!new:` creates the output before `FlexTrainLogger` can
check whether it is enabled.

For example:

```bash
python train.py hparams.yaml \
    --logging_wandb_enabled=True \
    --logging_wandb_project=my_project \
    --logging_wandb_name=experiment_001
```

### Aggregate Multiple Loggers

```python
from flexlog.flex import FlexTrainLogger
from flexlog.file import FlexFileTrainLogger

file_logger = FlexFileTrainLogger(save_file="logs/train.log")

logger = FlexTrainLogger(
    outputs={"file": file_logger},
    enabled={"file": True},
)

logger.log_stats(
    stats_meta={"epoch": 1},
    train_stats={"loss": 0.42},
    valid_stats={"loss": 0.51},
)
```

`FlexTrainLogger` forwards `log_stats` to every enabled logger. Artifact
methods are forwarded only to enabled loggers that support the matching method:

- `log_image(...)`
- `log_text(...)`
- `log_audio(...)`

This means a logger can support only text, only audio, or any other subset of
artifact methods.

### Enable or Disable Outputs

```python
logger = FlexTrainLogger(
    outputs={
        "file": file_logger,
        "wandb": wandb_logger,
    },
    enabled={
        "file": True,
        "wandb": False,
    },
)
```

Disabled outputs are skipped for both stats and artifact logging.

## File Artifacts

`FlexFileTrainLogger` saves artifacts under a progress folder next to the train
log file by default:

```python
from speechbrain import Stage
import torch

from flexlog.file import FlexFileTrainLogger

logger = FlexFileTrainLogger(save_file="logs/train.log")

logger.log_text(
    key="transcript",
    text="hello world",
    stats_meta={"epoch": 3},
    stage=Stage.VALID,
)

logger.log_audio(
    key="sample",
    audio=torch.zeros(16000),
    sample_rate=16000,
    stats_meta={"epoch": 3},
    stage=Stage.VALID,
)
```

When `stage` and `epoch` are provided, artifacts are saved as:

```text
logs/progress/valid/3/transcript.txt
logs/progress/valid/3/sample.wav
```

If `stage` is omitted, the epoch folder is placed directly under
`progress/`. If `epoch` is omitted, artifacts are written directly under the
stage folder or the progress folder.

## Weights & Biases

`FlexWandBLogger` extends SpeechBrain's `WandBLogger` with artifact helpers:

```python
from flexlog.wandb import FlexWandBLogger

logger = FlexWandBLogger(...)

logger.save_text("caption", "hello", stats_meta={"epoch": 1})
logger.save_image("plot", image_array, stats_meta={"epoch": 1})
logger.log_audio("sample", audio_tensor, sample_rate=16000)
```

For file paths and W&B steps, FlexLog accepts the SpeechBrain epoch metadata
keys `epoch`, `Epoch`, and `Epoch loaded`.

## Development

Run tests with:

```bash
PYTHONPATH=src pytest tests -q
```

Some Torch/SpeechBrain environments need an explicit temporary directory:

```bash
mkdir -p .pytest-tmp
TMPDIR=$PWD/.pytest-tmp TEMP=$PWD/.pytest-tmp TMP=$PWD/.pytest-tmp \
TORCHINDUCTOR_CACHE_DIR=$PWD/.pytest-tmp/torchinductor \
PYTHONPATH=src pytest -p no:cacheprovider tests -q
rm -rf .pytest-tmp
```

## License

FlexLog is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE).
