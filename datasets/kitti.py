import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.base import VidValDataset, read_frame_paths

KB_CROP_HEIGHT = 352
KB_CROP_WIDTH = 1216
GARG_CROP_TOP = 0.40810811
GARG_CROP_BOTTOM = 0.99189189
GARG_CROP_LEFT = 0.03594771
GARG_CROP_RIGHT = 0.96405229
MIN_DEPTH = 1e-3
MAX_DEPTH = 80.0


def apply_kb_crop(array: np.ndarray, h: int, w: int) -> np.ndarray:
    top = h - KB_CROP_HEIGHT
    left = (w - KB_CROP_WIDTH) // 2
    return array[top : top + KB_CROP_HEIGHT, left : left + KB_CROP_WIDTH]


def apply_garg_mask(depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    mask = np.zeros_like(depth, dtype=bool)
    mask[
        int(GARG_CROP_TOP * h) : int(GARG_CROP_BOTTOM * h),
        int(GARG_CROP_LEFT * w) : int(GARG_CROP_RIGHT * w),
    ] = True
    return mask & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)


class KITTIVal(VidValDataset):
    def __init__(
        self,
        num_frames: int = 7,
        frame_size: int = 256,
        time_stride_seconds: float = 0.2,
        max_aspect_ratio: float = 2.0,
        fps: float = 10.0,
    ):
        super().__init__(
            num_frames=num_frames,
            frame_size=frame_size,
            time_stride_seconds=time_stride_seconds,
            max_aspect_ratio=max_aspect_ratio,
        )
        self.fps = fps

        root = Path(os.environ["KITTI_ROOT"])
        split_file = Path(__file__).parent / "kitti_eigen_test.txt"

        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2 or parts[1] == "None":
                    continue

                tokens = parts[0].split("/")
                date, drive, frame_name = tokens[0], tokens[1], tokens[-1]
                frame_num = int(Path(frame_name).stem)

                if frame_num < 19:
                    continue

                stride_frames = int(self.time_stride_seconds * self.fps)
                start_idx = frame_num - (self.num_frames - 1) * stride_frames
                if start_idx < 0:
                    continue

                frame_indices = list(range(start_idx, frame_num + 1, stride_frames))
                img_dir = root / date / drive / "image_02" / "data"
                depth_dir = self._find_depth_dir(root, drive)
                if depth_dir is None:
                    continue

                frame_paths = [img_dir / f"{idx:010d}.png" for idx in frame_indices]
                depth_paths = [depth_dir / f"{idx:010d}.png" for idx in frame_indices]
                self.samples.append(((frame_paths, frame_indices), depth_paths))
        self._shuffle_samples()

    @staticmethod
    def _find_depth_dir(root: Path, drive: str) -> Path | None:
        for split in ("val", "train"):
            d = root / split / drive / "proj_depth" / "groundtruth" / "image_02"
            if d.exists():
                return d
        return None

    def _get_frames_and_timestamps(self, source) -> tuple[np.ndarray, torch.Tensor]:
        frames, timestamps = read_frame_paths(*source, self.fps)
        h, w = frames[0].shape[:2]
        frames = np.stack([apply_kb_crop(frame, h, w) for frame in frames])
        return frames, timestamps

    def _load_labels(self, label_data: list[Path]) -> torch.Tensor:
        depths = [np.array(Image.open(path), dtype=np.float32) / 256.0 for path in label_data]
        h, w = depths[0].shape[:2]

        cropped_depths = []
        for depth in depths:
            depth_crop = apply_kb_crop(depth, h, w)
            valid_mask = apply_garg_mask(depth_crop)
            cropped_depths.append(torch.from_numpy(depth_crop * valid_mask))

        return torch.stack(cropped_depths)
