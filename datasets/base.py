import io
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import decord
import numpy as np
import torch
from decord import DECORDError
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms import RandomResizedCrop

DATA_ERRORS = (
    OSError,
    KeyError,
    IndexError,
    ValueError,
    RuntimeError,
    DECORDError,
)


def as_hw(frame_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(frame_size, int):
        return frame_size, frame_size
    h, w = frame_size
    return int(h), int(w)


def sample_frame_indices(
    timestamps: torch.Tensor,
    num_frames: int,
    time_stride_range: tuple[float, float],
    generator: torch.Generator | None = None,
) -> tuple[list[int], torch.Tensor]:
    min_dt, max_dt = time_stride_range
    deltas = (
        torch.rand(num_frames - 1, generator=generator) * (max_dt - min_dt) + min_dt
    )
    cumsum = torch.cat([torch.zeros(1), deltas.cumsum(0)])
    max_start = max(0, timestamps[-1].item() - cumsum[-1].item())
    tgt = torch.rand(1, generator=generator).item() * max_start + cumsum
    indices = (timestamps.unsqueeze(0) - tgt.unsqueeze(1)).abs().argmin(dim=1).tolist()
    return indices, timestamps[indices]


class VidReader(decord.VideoReader):
    @property
    def timestamps(self) -> torch.Tensor:
        return torch.tensor(
            [self.get_frame_timestamp(i)[0] for i in range(len(self))],
            dtype=torch.float32,
        )

    def get_batch(self, indices: list[int]) -> np.ndarray:
        return super().get_batch(indices).asnumpy()


class FrameReader:
    def __init__(
        self, timestamps: list[float], read_fn: callable
    ) -> None:
        self.timestamps = torch.as_tensor(timestamps, dtype=torch.float32)
        self._read = read_fn

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_batch(self, indices: list[int]) -> np.ndarray:
        return np.stack([np.array(self._read(i)) for i in indices])


def extract_frames_and_timestamps(
    vr: VidReader | FrameReader, frame_indices: list[int]
) -> tuple[np.ndarray, torch.Tensor]:
    return vr.get_batch(frame_indices), vr.timestamps[frame_indices]


def compute_middle_frame_indices(
    vr: VidReader | FrameReader, num_frames: int, time_stride_seconds: float
) -> list[int]:
    vid_len = len(vr)
    fps = (vid_len - 1) / vr.timestamps[-1].item()
    stride = max(1, round(time_stride_seconds * fps))

    total_span = (num_frames - 1) * stride
    start = max(0, vid_len // 2 - total_span // 2)
    start = min(start, vid_len - 1 - total_span)

    return [start + i * stride for i in range(num_frames)]


class TrainDataset(IterableDataset, ABC):
    samples: list[Any]
    exclude_list: str | None = None

    def __init__(self, **kw: Any) -> None:
        for key, value in kw.items():
            setattr(self, key, value)

    def _filter_excluded(self, candidates: list[Any]) -> list[Any]:
        if not self.exclude_list:
            return candidates
        excluded = {
            line.removesuffix(".mp4") for line in Path(self.exclude_list).read_text().splitlines()
        }
        result = []
        for candidate in candidates:
            key = "/".join(Path(candidate).parts[-2:]).removesuffix(".mp4")
            if key in excluded:
                excluded.discard(key)
            else:
                result.append(candidate)
        assert not excluded, f"Videos in {self.exclude_list} not found in dataset: {excluded}"
        return result

    def _len(self) -> int:
        return len(self.samples)

    def _augment(
        self, frames: np.ndarray, timestamps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = frames[0].shape[:2]
        out_h, out_w = as_hw(self.frame_size)
        tgt_ratio = out_w / out_h
        ratio = (tgt_ratio / self.ratio_jitter, tgt_ratio * self.ratio_jitter)
        top, left, ch, cw = RandomResizedCrop.get_params(
            Image.new("RGB", (w, h)), self.scale, ratio
        )
        batch = np.stack(
            [
                cv2.resize(frame[top : top + ch, left : left + cw], (out_w, out_h))
                for frame in frames
            ]
        )
        if self.horizontal_flip and random.random() < 0.5:
            batch = batch[:, :, ::-1].copy()
        return torch.from_numpy(batch).permute(0, 3, 1, 2), timestamps

    @abstractmethod
    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        num_items = self._len()
        if num_items == 0:
            raise ValueError("Dataset is empty")
        rng = torch.Generator()
        rng.seed()
        while True:
            try:
                yield self._get_sample(
                    torch.randint(num_items, (1,), generator=rng).item()
                )
            except DATA_ERRORS:
                continue


def read_frame_paths(
    frame_paths: list[str], frame_indices: list[int], fps: float
) -> tuple[np.ndarray, torch.Tensor]:
    vr = FrameReader(
        [i / fps for i in frame_indices],
        lambda i: Image.open(frame_paths[i]).convert("RGB"),
    )
    return vr.get_batch(list(range(len(vr)))), vr.timestamps


def create_vid_reader(source: str | tuple | io.BytesIO) -> VidReader | FrameReader:
    if isinstance(source, tuple):
        store, vid = source
        return store.reader(vid)
    if isinstance(source, str) and Path(source).is_dir():
        paths = sorted(str(path) for path in Path(source).glob("*.jpg"))
        timestamps = [float(Path(path).stem.split("_")[1]) for path in paths]
        return FrameReader(timestamps, lambda i: Image.open(paths[i]).convert("RGB"))
    return VidReader(source, num_threads=1)


class VidTrainDataset(TrainDataset, ABC):
    prefetcher: Any = None

    @abstractmethod
    def _get_source(self, idx: int) -> Any: ...

    def _load_frames(self, idx: int) -> tuple[np.ndarray, torch.Tensor]:
        if self.prefetcher:
            raw_frames, timestamps = self.prefetcher.get()
            return (
                np.stack(
                    [
                        np.array(Image.open(io.BytesIO(frame_bytes)).convert("RGB"))
                        for frame_bytes in raw_frames
                    ]
                ),
                timestamps,
            )
        vr = create_vid_reader(self._get_source(idx))
        frame_indices, timestamps = sample_frame_indices(
            vr.timestamps, self.num_frames, self.time_stride_range
        )
        return vr.get_batch(frame_indices), timestamps

    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames, timestamps = self._load_frames(idx)
        out, timestamps = self._augment(frames, timestamps)
        return out, timestamps - timestamps[0]


class ValDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...


def compute_resize_sizes(
    h: int,
    w: int,
    frame_size: int | tuple[int, int],
    max_aspect_ratio: float = 2.0,
) -> tuple[int, int]:
    if not isinstance(frame_size, int):
        return as_hw(frame_size)
    scale = frame_size / min(h, w)
    max_size = int(frame_size * max_aspect_ratio)
    return min(int(round(h * scale)), max_size), min(int(round(w * scale)), max_size)


class VidValDataset(ValDataset, ABC):
    samples: list[Any]

    def __init__(self, **kw: Any) -> None:
        for key, value in kw.items():
            setattr(self, key, value)
        self.samples = []

    def _shuffle_samples(self) -> None:
        """Shuffle so first N val samples are diverse for plotting."""
        random.Random(0).shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_frames_and_timestamps(
        self, source: Any
    ) -> tuple[np.ndarray, torch.Tensor]:
        vr = create_vid_reader(source)
        indices = compute_middle_frame_indices(
            vr, self.num_frames, self.time_stride_seconds
        )
        return extract_frames_and_timestamps(vr, indices)

    def _load_labels(self, label_data: Any) -> torch.Tensor | None:
        return None

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        item = self.samples[idx]
        source, label_data = item if isinstance(item, tuple) else (item, None)
        frames, timestamps = self._get_frames_and_timestamps(source)
        labels = self._load_labels(label_data)
        h, w = frames[0].shape[:2]
        new_h, new_w = compute_resize_sizes(
            h, w, self.frame_size, self.max_aspect_ratio
        )
        frames = np.stack([cv2.resize(frame, (new_w, new_h)) for frame in frames])
        return (
            torch.from_numpy(frames).permute(0, 3, 1, 2),
            timestamps - timestamps[0],
            labels,
            idx,
        )
