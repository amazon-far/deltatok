#!/usr/bin/env python3
"""Download the small official KITTI subset used by the Apple M4 validation guide."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZipInfo

from remotezip import RemoteZip


DRIVE = "2011_09_26_drive_0002_sync"
RAW_URL = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    "2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip"
)
DEPTH_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"


def selected_members(archive: RemoteZip, marker: str) -> list[ZipInfo]:
    return [
        info
        for info in archive.infolist()
        if marker in info.filename and not info.is_dir()
    ]


def extract_missing(archive: RemoteZip, members: list[ZipInfo], root: Path) -> int:
    downloaded = 0
    for index, info in enumerate(members, 1):
        relative = PurePosixPath(info.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe archive member: {info.filename}")

        destination = root.joinpath(*relative.parts)
        if destination.is_file() and destination.stat().st_size == info.file_size:
            continue

        archive.extract(info, root)
        downloaded += 1
        if index % 10 == 0 or index == len(members):
            print(f"  processed {index}/{len(members)} files", flush=True)
    return downloaded


def download_archive_members(
    url: str, marker: str, root: Path, label: str
) -> tuple[int, int]:
    print(f"Inspecting official KITTI {label} archive...", flush=True)
    with RemoteZip(url) as archive:
        members = selected_members(archive, marker)
        if not members:
            raise RuntimeError(f"No {label} members matched {marker!r}")
        print(f"Found {len(members)} {label} files.", flush=True)
        downloaded = extract_missing(archive, members, root)
    return len(members), downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_data/kitti"),
        help="KITTI root directory to create (default: validation_data/kitti)",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    rgb_total, rgb_downloaded = download_archive_members(
        RAW_URL, f"{DRIVE}/image_02/data/", root, "RGB"
    )
    depth_total, depth_downloaded = download_archive_members(
        DEPTH_URL,
        f"{DRIVE}/proj_depth/groundtruth/image_02/",
        root,
        "depth",
    )

    print(f"KITTI subset ready at {root}")
    print(
        f"RGB: {rgb_total} total ({rgb_downloaded} downloaded); "
        f"depth: {depth_total} total ({depth_downloaded} downloaded)"
    )


if __name__ == "__main__":
    main()
