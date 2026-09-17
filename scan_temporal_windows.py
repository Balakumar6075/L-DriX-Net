import csv
import glob
import os
import re

import numpy as np


BASE_DIR = r"dataset\Subject01_1_data"

TEMPORAL_LENGTH = 16

OUTPUT_FILE = "temporal_window_features.csv"


def parse_gaze_file(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # --------------------------------------------------------
    # Gaze location
    # --------------------------------------------------------

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

    gaze_x = values[0]
    gaze_y = values[1]

    # --------------------------------------------------------
    # Left gaze direction
    # --------------------------------------------------------

    match = re.search(
        r"Left_Gaze_Dir:\s*\[([^\]]+)\]",
        text
    )

    left = None

    if match is not None:

        values = re.split(
            r"[\s,]+",
            match.group(1).strip()
        )

        values = [
            float(v)
            for v in values
            if v
        ]

        if len(values) == 3:
            left = np.array(
                values,
                dtype=np.float32
            )

    # --------------------------------------------------------
    # Right gaze direction
    # --------------------------------------------------------

    match = re.search(
        r"Right_Gaze_Dir:\s*\[([^\]]+)\]",
        text
    )

    right = None

    if match is not None:

        values = re.split(
            r"[\s,]+",
            match.group(1).strip()
        )

        values = [
            float(v)
            for v in values
            if v
        ]

        if len(values) == 3:
            right = np.array(
                values,
                dtype=np.float32
            )

    # --------------------------------------------------------
    # Combine gaze directions
    # --------------------------------------------------------

    if left is not None and right is not None:

        direction = (
            left + right
        ) / 2.0

    elif left is not None:

        direction = left

    elif right is not None:

        direction = right

    else:

        direction = None

    return {
        "x": gaze_x,
        "y": gaze_y,
        "direction": direction
    }


def main():

    print("=" * 70)
    print("Full Temporal Window Analysis")
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
    # Read all frame-level data once
    # --------------------------------------------------------

    frames = []

    for gaze_file in files:

        filename = os.path.basename(
            gaze_file
        )

        frame_id = int(
            filename[:8]
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

        if not os.path.exists(depth_file):
            continue

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

        frames.append(
            {
                "frame_id": frame_id,
                "x": data["x"],
                "y": data["y"],
                "depth": gaze_depth,
                "direction": data["direction"]
            }
        )

    print()
    print(
        "Valid synchronized frames:",
        len(frames)
    )

    # --------------------------------------------------------
    # Calculate features for every temporal window
    # --------------------------------------------------------

    rows = []

    total_windows = (
        len(frames)
        - TEMPORAL_LENGTH
        + 1
    )

    print(
        "Temporal windows:",
        total_windows
    )

    for start in range(total_windows):

        window = frames[
            start:start + TEMPORAL_LENGTH
        ]

        x = np.array(
            [f["x"] for f in window],
            dtype=np.float32
        )

        y = np.array(
            [f["y"] for f in window],
            dtype=np.float32
        )

        depth = np.array(
            [f["depth"] for f in window],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Gaze movement
        # ----------------------------------------------------

        dx = np.diff(x)

        dy = np.diff(y)

        gaze_movement_x = float(
            np.sum(np.abs(dx))
        )

        gaze_movement_y = float(
            np.sum(np.abs(dy))
        )

        total_gaze_movement = float(
            np.sum(
                np.sqrt(
                    dx ** 2 +
                    dy ** 2
                )
            )
        )

        # ----------------------------------------------------
        # Gaze stability
        # ----------------------------------------------------

        gaze_x_std = float(
            np.std(x)
        )

        gaze_y_std = float(
            np.std(y)
        )

        # ----------------------------------------------------
        # Gaze range
        # ----------------------------------------------------

        gaze_x_range = float(
            np.max(x) -
            np.min(x)
        )

        gaze_y_range = float(
            np.max(y) -
            np.min(y)
        )

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        depth_mean = float(
            np.mean(depth)
        )

        depth_std = float(
            np.std(depth)
        )

        depth_range = float(
            np.max(depth) -
            np.min(depth)
        )

        # ----------------------------------------------------
        # Gaze direction movement
        # ----------------------------------------------------

        directions = [
            f["direction"]
            for f in window
            if f["direction"] is not None
        ]

        direction_change = 0.0

        if len(directions) >= 2:

            directions = np.array(
                directions
            )

            direction_diffs = (
                np.diff(
                    directions,
                    axis=0
                )
            )

            direction_change = float(
                np.sum(
                    np.linalg.norm(
                        direction_diffs,
                        axis=1
                    )
                )
            )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        rows.append(
            {
                "window_index": start,

                "start_frame":
                    window[0]["frame_id"],

                "end_frame":
                    window[-1]["frame_id"],

                "gaze_movement_x":
                    gaze_movement_x,

                "gaze_movement_y":
                    gaze_movement_y,

                "total_gaze_movement":
                    total_gaze_movement,

                "gaze_x_std":
                    gaze_x_std,

                "gaze_y_std":
                    gaze_y_std,

                "gaze_x_range":
                    gaze_x_range,

                "gaze_y_range":
                    gaze_y_range,

                "depth_mean":
                    depth_mean,

                "depth_std":
                    depth_std,

                "depth_range":
                    depth_range,

                "direction_change":
                    direction_change
            }
        )

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    if rows:

        fieldnames = list(
            rows[0].keys()
        )

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Print summary statistics
    # --------------------------------------------------------

    feature_names = [
        "total_gaze_movement",
        "gaze_x_std",
        "gaze_y_std",
        "gaze_x_range",
        "gaze_y_range",
        "depth_mean",
        "depth_std",
        "depth_range",
        "direction_change"
    ]

    print()
    print("FEATURE RANGES")
    print("-" * 70)

    for feature in feature_names:

        values = np.array(
            [
                row[feature]
                for row in rows
            ]
        )

        print(
            f"{feature:<25}"
            f"min={values.min():.4f}   "
            f"max={values.max():.4f}   "
            f"mean={values.mean():.4f}"
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()