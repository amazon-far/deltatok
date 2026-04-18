import os
from pathlib import Path

from datasets.base import VidTrainDataset
from datasets.storage import ZipFrameStore, ZipPrefetcher, load_local_paths


class KineticsTrain(VidTrainDataset):
    def __init__(
        self,
        exclude_list: str = "datasets/kinetics_excluded.txt",
        num_frames: int = 8,
        frame_size: int = 256,
        time_stride_range: tuple[float, float] = (1 / 25, 1 / 3),
        horizontal_flip: bool = True,
        ratio_jitter: float = 4 / 3,
        scale: tuple[float, float] = (0.6, 1.0),
    ):
        super().__init__(
            exclude_list=exclude_list,
            num_frames=num_frames,
            frame_size=frame_size,
            time_stride_range=time_stride_range,
            horizontal_flip=horizontal_flip,
            ratio_jitter=ratio_jitter,
            scale=scale,
        )
        root = os.environ["KINETICS_ROOT"]
        if root.endswith(".zip"):
            # Pre-extracted frames in a zip archive (optionally split into .part* files)
            self.store = ZipFrameStore(root)
            candidates = [vid for vid in self.store.vids if "train" in Path(vid).parts]
            self.samples = self._filter_excluded(candidates)
            self.prefetcher = ZipPrefetcher(
                self.store, self.samples, num_frames, time_stride_range
            )
        else:
            # Local directory: train/<class>/<video>/ (frame dirs) or
            # train/<class>/<video>.mp4 (video files)
            self.store = None
            first_item = next(next(Path(root, "train").iterdir()).iterdir())
            if first_item.is_dir():
                candidates = sorted(
                    set(
                        str(Path(path).parent)
                        for path in load_local_paths(root, "train/**/*.jpg")
                    )
                )
            else:
                candidates = load_local_paths(root, "train/**/*.mp4")
            self.samples = self._filter_excluded(candidates)

    def _get_source(self, idx: int) -> tuple[ZipFrameStore, str] | str:
        return (self.store, self.samples[idx]) if self.store else self.samples[idx]
