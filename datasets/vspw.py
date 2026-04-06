from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.base import VidValDataset, read_frame_paths


class VSPWVal(VidValDataset):
    def __init__(
        self,
        root: str,
        num_frames: int = 7,
        frame_size: int = 256,
        time_stride_seconds: float = 0.2,
        max_aspect_ratio: float = 2.0,
        fps: float = 15.0,
        eval_frame_interval: int = 20,
    ):
        super().__init__(
            num_frames=num_frames,
            frame_size=frame_size,
            time_stride_seconds=time_stride_seconds,
            max_aspect_ratio=max_aspect_ratio,
        )
        self.fps = fps
        self.eval_frame_interval = eval_frame_interval

        root = Path(root)
        split_file = root / "val.txt"
        with open(split_file, "r") as f:
            vid_names = [line.strip() for line in f if line.strip()]

        data_dir = root / "data"

        for vid_name in vid_names:
            vid_path = data_dir / vid_name
            origin_path = vid_path / "origin"
            labels_path = vid_path / "mask"

            img_files = sorted(
                path for path in origin_path.glob("*.jpg") if not path.name.startswith("._")
            )
            stride_frames = int(self.time_stride_seconds * self.fps)

            label_indices = list(
                range(
                    self.eval_frame_interval - 1,
                    len(img_files),
                    self.eval_frame_interval,
                )
            )

            for label_idx in label_indices:
                end_idx = label_idx
                start_idx = end_idx - (self.num_frames - 1) * stride_frames
                if start_idx < 0:
                    continue

                frame_indices = list(range(start_idx, end_idx + 1, stride_frames))
                frame_paths = [str(img_files[i]) for i in frame_indices]
                label_paths = [
                    labels_path / img_files[i].name.replace(".jpg", ".png")
                    for i in frame_indices
                ]
                self.samples.append(((frame_paths, frame_indices), label_paths))
        self._shuffle_samples()

    def _get_frames_and_timestamps(self, source) -> tuple[np.ndarray, torch.Tensor]:
        return read_frame_paths(*source, self.fps)

    def _load_labels(self, label_data: list[Path]) -> torch.Tensor:
        labels_list = []
        for label_path in label_data:
            labels_frame = (
                torch.from_numpy(np.array(Image.open(label_path), dtype=np.int64)) - 1
            )
            labels_frame[(labels_frame < 0) | (labels_frame >= 124)] = 255
            labels_list.append(labels_frame)
        return torch.stack(labels_list)
