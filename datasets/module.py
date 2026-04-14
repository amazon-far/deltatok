from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets.base import TrainDataset, ValDataset, as_hw


def val_collate_fn(batch: list[tuple]) -> tuple[torch.Tensor, ...]:
    frames, timestamps, labels, sample_indices = zip(*batch)
    return (
        torch.stack(frames),
        torch.stack(timestamps),
        None if labels[0] is None else torch.stack(labels),
        torch.tensor(sample_indices),
    )


def _load_cls(cls_path: str) -> type[TrainDataset | ValDataset]:
    module_path, cls_name = cls_path.rsplit(".", 1)
    return getattr(__import__(module_path, fromlist=[cls_name]), cls_name)


class DataModule(LightningDataModule):
    train_dataset: TrainDataset
    val_datasets: dict[str, ValDataset]

    def __init__(
        self,
        val_datasets_cfg: dict[str, dict[str, Any]] | None = None,
        num_workers: int = 16,
        prefetch_factor: int | None = None,
        train_dataset_cfg: dict[str, Any] | None = None,
        batch_size: int = 128,
        frame_size: int | tuple[int, int] = 256,
    ) -> None:
        super().__init__()
        self.val_datasets_cfg = val_datasets_cfg
        self.train_dataset_cfg = train_dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.frame_size = (
            frame_size if isinstance(frame_size, int) else as_hw(frame_size)
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset_cfg is not None:
            cls = _load_cls(self.train_dataset_cfg["class_path"])
            self.train_dataset = cls(
                **self.train_dataset_cfg["init_args"], frame_size=self.frame_size
            )

        self.val_datasets = {}
        for name, cfg in (self.val_datasets_cfg or {}).items():
            cls = _load_cls(cfg["class_path"])
            self.val_datasets[name] = cls(
                **cfg["init_args"], frame_size=self.frame_size
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="fork" if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> list[DataLoader]:
        dataloaders = []
        for name, dataset in self.val_datasets.items():
            loader = DataLoader(
                dataset,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
                collate_fn=val_collate_fn,
                multiprocessing_context="fork" if self.num_workers > 0 else None,
            )
            loader.dataset_name = name
            dataloaders.append(loader)
        return dataloaders
