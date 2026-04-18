import logging
import sys
import warnings
from datetime import datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
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
    """Capture everything needed to reproduce the run."""

    def setup(self, trainer: Trainer, pl_module: Base, stage: str) -> None:
        if trainer.logger is not None:
            trainer.logger.experiment.log_code(
                ".", include_fn=lambda filename: filename.endswith((".py", ".yaml", ".env"))
            )

    def on_train_start(self, trainer: Trainer, pl_module: Base) -> None:
        if not trainer.is_global_zero or trainer.logger is None:
            return
        experiment = trainer.logger.experiment

        if Path(experiment.dir).exists():
            pkgs = sorted(f"{d.name}=={d.version}" for d in distributions())
            pkg_path = Path(experiment.dir) / f"pip-freeze-step{trainer.global_step}.txt"
            pkg_path.write_text("\n".join(pkgs))
            experiment.save(str(pkg_path), policy="now")

        entry = f"[step {trainer.global_step}] {datetime.now().isoformat()}\n$ {' '.join(sys.argv)}"
        experiment.notes = f"{experiment.notes or ''}\n{entry}".strip()


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args: Any, **kw: Any) -> None:
        load_dotenv()
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
