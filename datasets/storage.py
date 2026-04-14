import bisect
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


class SplitFile:
    """Seekable read-only file that may consist of one or more parts on disk.

    Handles both ``path`` (single file) and ``path.part000``, ``path.part001``, …
    (split file) transparently.
    """

    def __init__(self, path: str):
        self._parts = (
            [path] if os.path.exists(path) else
            sorted(str(p) for p in Path(path).parent.glob(Path(path).name + ".part*"))
        )
        if not self._parts:
            raise FileNotFoundError(path)
        self._sizes = [os.path.getsize(p) for p in self._parts]
        self._cum = [0]
        for s in self._sizes:
            self._cum.append(self._cum[-1] + s)
        self._pos, self._fhs, self._fds = 0, None, None

    def _spans(self, offset, size):
        while size > 0:
            idx = bisect.bisect_right(self._cum, offset) - 1
            local = offset - self._cum[idx]
            n = min(size, self._sizes[idx] - local)
            yield idx, local, n
            offset += n
            size -= n

    def seek(self, offset, whence=0):
        if whence == 0: self._pos = offset
        elif whence == 1: self._pos += offset
        elif whence == 2: self._pos = self._cum[-1] + offset
        return self._pos

    def tell(self): return self._pos
    def seekable(self): return True
    def readable(self): return True

    def read(self, size=-1):
        if size < 0: size = self._cum[-1] - self._pos
        if size <= 0: return b""
        if not self._fhs:
            self._fhs = [open(p, "rb") for p in self._parts]
        bufs = []
        for idx, local, n in self._spans(self._pos, size):
            self._fhs[idx].seek(local)
            bufs.append(self._fhs[idx].read(n))
        self._pos += size
        return b"".join(bufs)

    def close(self):
        for fh in self._fhs or []: fh.close()
        for fd in self._fds or []: os.close(fd)
        self._fhs = self._fds = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    def pread(self, size, offset):
        if not self._fds:
            self._fds = [os.open(p, os.O_RDONLY) for p in self._parts]
        return b"".join(
            os.pread(self._fds[idx], n, local)
            for idx, local, n in self._spans(offset, size)
        )


class ZipReader:
    def __init__(self, path: str, offsets: np.ndarray):
        self.path, self.offsets, self._fh = path, offsets, None

    def read(self, i: int) -> bytes:
        off, size, _, method = self.offsets[i]
        if not self._fh:
            self._fh = SplitFile(self.path)
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
    with SplitFile(path) as sf, zipfile.ZipFile(sf) as zf:
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
            self._readers[wid] = ZipReader(self.path, self._offsets)
        zip_reader = self._readers[wid]

        return FrameReader(
            timestamps,
            lambda i: Image.open(io.BytesIO(zip_reader.read(start + i))).convert("RGB"),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state


def _pread_frame(sf: SplitFile, offsets: np.ndarray, idx: int) -> bytes:
    off, size, _, method = offsets[idx]
    hdr = sf.pread(30, int(off))
    n = int.from_bytes(hdr[26:28], "little")
    x = int.from_bytes(hdr[28:30], "little")
    data = sf.pread(int(size), int(off) + 30 + n + x)
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
    sf = SplitFile(zip_path)

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
                        [_pread_frame(sf, offsets, start + i) for i in indices],
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
