import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from lightning import Trainer
from lightning.pytorch import cli
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

from training.base import Base


class LogRun(Callback):
    def setup(self, trainer: Trainer, pl_module: Base, stage: str) -> None:
        if trainer.logger is not None and hasattr(
            trainer.logger.experiment, "log_code"
        ):
            trainer.logger.experiment.log_code(
                ".",
                include_fn=lambda filename: filename.endswith((".py", ".yaml")),
            )

    def on_train_start(self, trainer: Trainer, pl_module: Base) -> None:
        if not trainer.is_global_zero or trainer.logger is None:
            return

        experiment = trainer.logger.experiment
        if not hasattr(experiment, "dir") or experiment.dir is None:
            return

        step = trainer.global_step

        metadata_path = Path(experiment.dir) / "wandb-metadata.json"
        snapshot = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        snapshot |= {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "command": sys.argv,
            "config": dict(experiment.config),
        }
        file_name = f"session_{step}.json"
        (Path(experiment.dir) / file_name).write_text(
            json.dumps(snapshot, indent=2, default=str)
        )
        experiment.save(file_name, policy="now")

        entry = f"[step {step}] {snapshot['timestamp']}\n$ {' '.join(sys.argv)}"
        notes = experiment.notes or ""
        experiment.notes = f"{notes}\n{entry}".strip()


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args: Any, **kw: Any) -> None:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        logging.getLogger().setLevel(logging.INFO)
        for msg in [
            r".*isinstance.*LeafSpec.*is deprecated.*",
            r".*functools.partial will be a method descriptor in future Python versions*",
            r".*No device id is provided via `init_process_group` or `barrier.*",
            r".*Precision bf16-mixed is not supported by the model summary.*",
            r".*Dynamo detected a call to a `functools.lru_cache`-wrapped function.*",
            r".*Trying to infer the `batch_size` from an ambiguous collection.*",
            r".*Found .* module\(s\) in eval mode at the start of training.*",
        ]:
            warnings.filterwarnings("ignore", message=msg)

        super().__init__(*args, **kw)


def cli_main() -> None:
    LightningCLI(
        Base,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "max_epochs": -1,
            "devices": 8,
            "enable_model_summary": False,
            "check_val_every_n_epoch": None,
            "val_check_interval": 5000,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="step"),
                LogRun(),
                ModelCheckpoint(
                    every_n_train_steps=50000,
                    save_top_k=-1,
                ),
                ModelCheckpoint(
                    monitor="losses/val",
                    mode="min",
                    save_top_k=1,
                ),
                ModelCheckpoint(
                    every_n_train_steps=500,
                    save_last=True,
                    save_top_k=0,
                    enable_version_counter=False,
                ),
            ],
            "log_every_n_steps": 1,
            "logger": {
                "class_path": "lightning.pytorch.loggers.wandb.WandbLogger",
                "init_args": {
                    "project": "deltatok",
                    "save_dir": str(Path(__file__).resolve().parent.parent / "runs"),
                },
            },
            "precision": "bf16-mixed",
        },
    )


if __name__ == "__main__":
    cli_main()
