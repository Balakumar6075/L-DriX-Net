import glob
import os
import re

import numpy as np


BASE_DIR = r"dataset\Subject01_1_data"


def get_gaze_location(gaze_file):

    with open(
        gaze_file,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    match = re.search(
        r"Gaze_Loc_2D:\s*\[([^\]]+)\]",
        text
    )

    if match is None:
        return None

    values = re.split(
        r"[\s,]+",
        match.group(1).strip()
    )

    values = [
        float(v)
        for v in values
        if v
    ]

    if len(values) != 2:
        return None

    x = int(round(values[0]))
    y = int(round(values[1]))

    return x, y


def main():

    print("=" * 60)
    print("Gaze → Depth Correspondence Test")
    print("=" * 60)

    gaze_dir = os.path.join(
        BASE_DIR,
        "gaze_info"
    )

    depth_dir = os.path.join(
        BASE_DIR,
        "scene_depth"
    )

    files = sorted(
        glob.glob(
            os.path.join(
                gaze_dir,
                "*_gaze.txt"
            )
        )
    )

    print()
    print("Testing first 10 synchronized samples")
    print()

    print(
        f"{'Frame':<10}"
        f"{'Gaze X':<10}"
        f"{'Gaze Y':<10}"
        f"{'Depth':<15}"
    )

    print("-" * 45)

    successful = 0

    for gaze_file in files[:10]:

        filename = os.path.basename(
            gaze_file
        )

        frame_id = int(
            filename[:8]
        )

        gaze = get_gaze_location(
            gaze_file
        )

        if gaze is None:
            print(
                frame_id,
                "Could not parse gaze"
            )
            continue

        x, y = gaze

        depth_file = os.path.join(
            depth_dir,
            f"{frame_id:08d}_depth.npy"
        )

        if not os.path.exists(depth_file):
            print(
                frame_id,
                "Depth file missing"
            )
            continue

        depth = np.load(
            depth_file
        )

        # ----------------------------------------------------
        # Check coordinates
        # ----------------------------------------------------

        height, width = depth.shape

        if not (
            0 <= x < width
            and
            0 <= y < height
        ):
            print(
                frame_id,
                "Gaze coordinate outside depth image"
            )
            continue

        depth_value = float(
            depth[y, x]
        )

        print(
            f"{frame_id:<10}"
            f"{x:<10}"
            f"{y:<10}"
            f"{depth_value:<15.4f}"
        )

        successful += 1

    print()
    print("-" * 45)
    print(
        "Successful:",
        successful,
        "/",
        min(10, len(files))
    )

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()