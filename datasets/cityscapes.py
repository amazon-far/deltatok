from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.base import VidValDataset, read_frame_paths

FPS = 16.0

_LABEL_TO_TRAIN = np.full(256, 255, dtype=np.int64)
for _id, _train in {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}.items():
    _LABEL_TO_TRAIN[_id] = _train


class CityscapesVal(VidValDataset):
    def __init__(
        self,
        root: str,
        num_frames: int = 7,
        frame_size: int = 256,
        time_stride_seconds: float = 0.1875,
        max_aspect_ratio: float = 2.0,
        fps: float = FPS,
    ):
        super().__init__(
            num_frames=num_frames,
            frame_size=frame_size,
            time_stride_seconds=time_stride_seconds,
            max_aspect_ratio=max_aspect_ratio,
        )
        self.fps = fps

        root = Path(root)
        self.frames_dir = root / "leftImg8bit_sequence" / "val"
        label_dir = root / "gtFine" / "val"
        stride_frames = int(self.time_stride_seconds * self.fps)

        for label_path in sorted(label_dir.rglob("*_gtFine_labelIds.png")):
            city, seq_id, frame_num = label_path.stem.replace(
                "_gtFine_labelIds", ""
            ).split("_")
            frame_num = int(frame_num)
            start = frame_num - (self.num_frames - 1) * stride_frames

            frame_indices = list(range(start, frame_num + 1, stride_frames))
            frame_paths = [
                self.frames_dir
                / label_path.parent.name
                / f"{city}_{seq_id}_{i:06d}_leftImg8bit.png"
                for i in frame_indices
            ]
            self.samples.append(((frame_paths, frame_indices), label_path))

    def _get_frames_and_timestamps(self, source) -> tuple[np.ndarray, torch.Tensor]:
        return read_frame_paths(*source, self.fps)

    def _load_labels(self, label_data: Path) -> torch.Tensor:
        label_ids = np.array(Image.open(label_data), dtype=np.int64)
        label = torch.as_tensor(_LABEL_TO_TRAIN[label_ids])
        labels = torch.full((self.num_frames, *label.shape), 255, dtype=torch.int64)
        labels[-1] = label
        return labels
