#!/usr/bin/env python3
"""Download the ten source videos into ./videos.

Run once after unpacking:

    python download_videos.py
"""

from __future__ import annotations

import os
import urllib.request

INTEL = ("https://raw.githubusercontent.com/intel-iot-devkit/"
         "sample-videos/master/")
OPENCV = ("https://raw.githubusercontent.com/opencv/opencv_extra/"
          "master/testdata/cv/video/768x576.avi")

FILES = [
    (INTEL + "person-bicycle-car-detection.mp4", "person-bicycle-car-detection.mp4"),
    (INTEL + "car-detection.mp4", "car-detection.mp4"),
    (INTEL + "people-detection.mp4", "people-detection.mp4"),
    (INTEL + "store-aisle-detection.mp4", "store-aisle-detection.mp4"),
    (INTEL + "worker-zone-detection.mp4", "worker-zone-detection.mp4"),
    (INTEL + "bottle-detection.mp4", "bottle-detection.mp4"),
    (INTEL + "bolt-detection.mp4", "bolt-detection.mp4"),
    (INTEL + "one-by-one-person-detection.mp4", "one-by-one-person-detection.mp4"),
    (INTEL + "fruit-and-vegetable-detection.mp4", "fruit-and-vegetable-detection.mp4"),
    (OPENCV, "street-pedestrians-768x576.avi"),
]


def main(out_dir: str = "videos") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, (url, name) in enumerate(FILES, 1):
        dest = os.path.join(out_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[{i}/{len(FILES)}] {name} already here, skipping")
            continue
        print(f"[{i}/{len(FILES)}] downloading {name} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:                      # noqa: BLE001
            print(f"    failed: {exc}")
            continue
        print(f"    {os.path.getsize(dest) / 1e6:.1f} MB")
    print(f"\ndone -> {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
