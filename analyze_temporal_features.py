import glob
import os
import re

import numpy as np


BASE_DIR = r"dataset\Subject01_1_data"


def parse_gaze_file(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # --------------------------------------------------------
    # 2D gaze location
    # --------------------------------------------------------

    match = re.search(
        r"Gaze_Loc_2D:\s*\[([^\]]+)\]",
        text
    )

    if match is None:
        return None

    gaze_values = re.split(
        r"[\s,]+",
        match.group(1).strip()
    )

    gaze_values = [
        float(v)
        for v in gaze_values
        if v
    ]

    if len(gaze_values) != 2:
        return None

    gaze_x = gaze_values[0]
    gaze_y = gaze_values[1]

    # --------------------------------------------------------
    # Left gaze direction
    # --------------------------------------------------------

    match = re.search(
        r"Left_Gaze_Dir:\s*\[([^\]]+)\]",
        text
    )

    if match is not None:

        values = re.split(
            r"[\s,]+",
            match.group(1).strip()
        )

        left_gaze = np.array(
            [
                float(v)
                for v in values
                if v
            ],
            dtype=np.float32
        )

    else:

        left_gaze = None

    # --------------------------------------------------------
    # Right gaze direction
    # --------------------------------------------------------

    match = re.search(
        r"Right_Gaze_Dir:\s*\[([^\]]+)\]",
        text
    )

    if match is not None:

        values = re.split(
            r"[\s,]+",
            match.group(1).strip()
        )

        right_gaze = np.array(
            [
                float(v)
                for v in values
                if v
            ],
            dtype=np.float32
        )

    else:

        right_gaze = None

    return {
        "x": gaze_x,
        "y": gaze_y,
        "left_gaze": left_gaze,
        "right_gaze": right_gaze
    }


def gaze_direction(left, right):

    if left is not None and right is not None:

        return (
            left + right
        ) / 2.0

    if left is not None:
        return left

    if right is not None:
        return right

    return None


def main():

    print("=" * 70)
    print("Temporal Driver Behavior Feature Analysis")
    print("=" * 70)

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

    # --------------------------------------------------------
    # Analyze first 16 synchronized samples
    # --------------------------------------------------------

    samples = []

    for gaze_file in files[:16]:

        frame_id = int(
            os.path.basename(
                gaze_file
            )[:8]
        )

        data = parse_gaze_file(
            gaze_file
        )

        if data is None:
            continue

        depth_file = os.path.join(
            depth_dir,
            f"{frame_id:08d}_depth.npy"
        )

        depth = np.load(
            depth_file
        )

        x = int(round(data["x"]))
        y = int(round(data["y"]))

        height, width = depth.shape

        if not (
            0 <= x < width
            and
            0 <= y < height
        ):
            continue

        gaze_depth = float(
            depth[y, x]
        )

        direction = gaze_direction(
            data["left_gaze"],
            data["right_gaze"]
        )

        samples.append(
            {
                "frame": frame_id,
                "x": data["x"],
                "y": data["y"],
                "depth": gaze_depth,
                "direction": direction
            }
        )

    # --------------------------------------------------------
    # Display frame-level values
    # --------------------------------------------------------

    print()
    print(
        f"{'Frame':<8}"
        f"{'Gaze X':<10}"
        f"{'Gaze Y':<10}"
        f"{'Depth':<12}"
        f"{'Dir X':<10}"
        f"{'Dir Y':<10}"
        f"{'Dir Z':<10}"
    )

    print("-" * 70)

    for sample in samples:

        direction = sample["direction"]

        if direction is not None:

            dx, dy, dz = direction

        else:

            dx = dy = dz = np.nan

        print(
            f"{sample['frame']:<8}"
            f"{sample['x']:<10.1f}"
            f"{sample['y']:<10.1f}"
            f"{sample['depth']:<12.3f}"
            f"{dx:<10.4f}"
            f"{dy:<10.4f}"
            f"{dz:<10.4f}"
        )

    # --------------------------------------------------------
    # Temporal statistics
    # --------------------------------------------------------

    x_values = np.array(
        [s["x"] for s in samples]
    )

    y_values = np.array(
        [s["y"] for s in samples]
    )

    depth_values = np.array(
        [s["depth"] for s in samples]
    )

    print()
    print("TEMPORAL STATISTICS")
    print("-" * 70)

    print(
        "Gaze X range:",
        f"{x_values.min():.2f}",
        "to",
        f"{x_values.max():.2f}"
    )

    print(
        "Gaze Y range:",
        f"{y_values.min():.2f}",
        "to",
        f"{y_values.max():.2f}"
    )

    print(
        "Gaze X movement:",
        f"{np.sum(np.abs(np.diff(x_values))):.2f}"
    )

    print(
        "Gaze Y movement:",
        f"{np.sum(np.abs(np.diff(y_values))):.2f}"
    )

    print(
        "Gaze position std:",
        f"{np.std(x_values):.2f},",
        f"{np.std(y_values):.2f}"
    )

    print(
        "Depth range:",
        f"{depth_values.min():.3f}",
        "to",
        f"{depth_values.max():.3f}"
    )

    print(
        "Depth mean:",
        f"{depth_values.mean():.3f}"
    )

    print(
        "Depth std:",
        f"{depth_values.std():.3f}"
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()