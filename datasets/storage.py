import hashlib
import io
import logging
import multiprocessing
import os
import random
import struct
import zipfile
import zlib
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from PIL import Image
from torch.utils.data import get_worker_info

from datasets.base import DATA_ERRORS, FrameReader, sample_frame_indices

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(key: str, suffix: str = ".txt") -> Path:
    return CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}{suffix}"


def _cached_lines(key: str, fetch: callable) -> list[str]:
    path = _cache_path(key)
    if path.exists():
        return path.read_text().splitlines()
    lines = fetch()
    path.write_text("\n".join(lines))
    return lines


def load_local_paths(root: str, pattern: str) -> list[str]:
    return _cached_lines(
        f"{root}:{pattern}", lambda: [str(path) for path in sorted(Path(root).glob(pattern))]
    )


class DirectZipReader:
    def __init__(self, path: str, offsets: np.ndarray):
        self.path, self.offsets, self._fh = path, offsets, None

    def read(self, i: int) -> bytes:
        off, size, _, method = self.offsets[i]
        if not self._fh:
            self._fh = open(self.path, "rb")
        self._fh.seek(off + 26)
        n, x = struct.unpack("<HH", self._fh.read(4))
        self._fh.seek(n + x, 1)
        data = self._fh.read(size)
        return data if method == 0 else zlib.decompress(data, -15)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def __del__(self):
        if self._fh:
            self._fh.close()


def _load_zip_index(path: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    cache = _cache_path(path, "_combined.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return (
            list(d["vids"]),
            d["vid_info"],
            d["frame_offsets"],
            d["frame_timestamps"],
        )

    logger.info("Building zip index for %s (one-time, ~15 min)...", path)
    with zipfile.ZipFile(path) as zf:
        infos = sorted(
            (
                (
                    info.filename,
                    info.header_offset,
                    info.compress_size,
                    info.file_size,
                    info.compress_type,
                )
                for info in zf.infolist()
                if info.filename.endswith(".jpg")
            ),
            key=lambda x: x[0],
        )

    vids, vid_info, frame_offsets, frame_timestamps, prev_vid, start = (
        [],
        [],
        [],
        [],
        None,
        0,
    )
    for i, (name, *off) in enumerate(infos):
        vid = str(Path(name).parent)
        if vid != prev_vid:
            if prev_vid:
                vids.append(prev_vid)
                vid_info.append((start, i))
            prev_vid, start = vid, i
        frame_offsets.append(off)
        frame_timestamps.append(float(Path(name).stem.split("_")[1]))
    vids.append(prev_vid)
    vid_info.append((start, len(infos)))

    np.savez(
        cache,
        vids=np.array(vids, dtype=object),
        vid_info=np.array(vid_info, dtype=np.int64),
        frame_offsets=np.array(frame_offsets, dtype=np.int64),
        frame_timestamps=np.array(frame_timestamps, dtype=np.float32),
    )
    return (
        vids,
        np.array(vid_info, dtype=np.int64),
        np.array(frame_offsets, dtype=np.int64),
        np.array(frame_timestamps, dtype=np.float32),
    )


class ZipFrameStore:
    def __init__(self, path: str):
        self.path = path
        self.vids, self._vid_info, self._offsets, self._timestamps = _load_zip_index(
            path
        )
        self._vid_index = {vid: i for i, vid in enumerate(self.vids)}
        self._readers = {}

    def reader(self, vid: str):
        start, end = self._vid_info[self._vid_index[vid]]
        timestamps = self._timestamps[start:end]

        wid = (w := get_worker_info()) and w.id
        if wid not in self._readers:
            self._readers[wid] = DirectZipReader(self.path, self._offsets)
        zip_reader = self._readers[wid]

        return FrameReader(
            timestamps,
            lambda i: Image.open(io.BytesIO(zip_reader.read(start + i))).convert("RGB"),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state


def _pread_frame(fd: int, offsets: np.ndarray, idx: int) -> bytes:
    off, size, _, method = offsets[idx]
    chunk = os.pread(fd, int(30 + 300 + size), int(off))
    n = int.from_bytes(chunk[26:28], "little")
    x = int.from_bytes(chunk[28:30], "little")
    data = chunk[30 + n + x : 30 + n + x + int(size)]
    return data if method == 0 else zlib.decompress(data, -15)


def _prefetch_loop(
    zip_path,
    offsets,
    timestamps,
    sample_ranges,
    num_frames,
    time_stride_range,
    queue,
    num_threads,
):
    fd = os.open(zip_path, os.O_RDONLY)

    def worker() -> None:
        rng = random.Random()
        gen = torch.Generator()
        gen.seed()
        while True:
            try:
                start, end = sample_ranges[rng.randrange(len(sample_ranges))]
                range_timestamps = torch.as_tensor(timestamps[start:end])
                indices, sampled_timestamps = sample_frame_indices(
                    range_timestamps, num_frames, time_stride_range, gen
                )
                queue.put(
                    (
                        [_pread_frame(fd, offsets, start + i) for i in indices],
                        sampled_timestamps,
                    )
                )
            except DATA_ERRORS:
                continue

    for _ in range(num_threads):
        t = Thread(target=worker, daemon=True)
        t.start()
    t.join()


class ZipPrefetcher:
    def __init__(
        self,
        store: ZipFrameStore,
        vids: list[str],
        num_frames: int,
        time_stride_range: tuple[float, float],
        num_threads: int = 8,
        queue_maxsize: int = 256,
    ):
        sample_ranges = np.array(
            [store._vid_info[store._vid_index[vid]] for vid in vids], dtype=np.int64
        )
        self._queue = multiprocessing.Queue(maxsize=queue_maxsize)
        ctx = multiprocessing.get_context("fork")
        self._process = ctx.Process(
            target=_prefetch_loop,
            args=(
                store.path,
                store._offsets,
                store._timestamps,
                sample_ranges,
                num_frames,
                time_stride_range,
                self._queue,
                num_threads,
            ),
            daemon=True,
        )
        self._process.start()

    def get(self) -> tuple[list[bytes], torch.Tensor]:
        return self._queue.get()
