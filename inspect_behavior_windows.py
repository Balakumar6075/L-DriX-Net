import csv
import glob
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"dataset\Subject01_1_data"

FACE_DIR = os.path.join(
    BASE_DIR,
    "face_ims"
)

SCENE_DIR = os.path.join(
    BASE_DIR,
    "scene_ims"
)

GAZE_DIR = os.path.join(
    BASE_DIR,
    "gaze_info"
)

FEATURE_FILE = "temporal_window_features.csv"

OUTPUT_DIR = "behavior_candidates"

TEMPORAL_LENGTH = 16

WINDOWS_TO_SELECT = 5

COLUMNS = 4

THUMBNAIL_WIDTH = 300

THUMBNAIL_HEIGHT = 220

LABEL_HEIGHT = 35


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(directory, frame_id):

    # Face images:
    # 00000190_face.png
    #
    # Scene images:
    # 00000190_scene.png

    suffixes = [
        "_face",
        "_scene",
        ""
    ]

    extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".PNG",
        ".JPG",
        ".JPEG"
    ]

    for suffix in suffixes:

        for extension in extensions:

            path = os.path.join(
                directory,
                f"{frame_id:08d}{suffix}{extension}"
            )

            if os.path.exists(path):
                return path

    return None


# ============================================================
# LOAD SYNCHRONIZED FRAME IDS
# ============================================================

def get_synchronized_frame_ids():

    def get_ids(directory, suffix):

        paths = glob.glob(
            os.path.join(
                directory,
                f"*{suffix}"
            )
        )

        ids = set()

        for path in paths:

            filename = os.path.basename(path)

            try:

                frame_id = int(
                    filename[:8]
                )

                ids.add(frame_id)

            except ValueError:

                continue

        return ids

    face_ids = get_ids(
        FACE_DIR,
        ".jpg"
    )

    if not face_ids:

        face_ids = get_ids(
            FACE_DIR,
            ".png"
        )

    scene_ids = get_ids(
        SCENE_DIR,
        ".jpg"
    )

    if not scene_ids:

        scene_ids = get_ids(
            SCENE_DIR,
            ".png"
        )

    gaze_ids = get_ids(
        GAZE_DIR,
        "_gaze.txt"
    )

    synchronized = (
        face_ids
        & scene_ids
        & gaze_ids
    )

    return sorted(
        synchronized
    )


# ============================================================
# LOAD CSV
# ============================================================

def load_features():

    rows = []

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            numeric_fields = [
                "window_index",
                "start_frame",
                "end_frame",
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

            for field in numeric_fields:

                if field == "window_index":
                    row[field] = int(row[field])

                elif field in [
                    "start_frame",
                    "end_frame"
                ]:

                    row[field] = int(row[field])

                else:

                    row[field] = float(
                        row[field]
                    )

            rows.append(row)

    return rows


# ============================================================
# SELECT CANDIDATE WINDOWS
# ============================================================

def select_windows(rows):

    selected = {}

    selected["stable"] = sorted(
        rows,
        key=lambda r:
        r["total_gaze_movement"]
    )[
        :WINDOWS_TO_SELECT
    ]

    selected["high_movement"] = sorted(
        rows,
        key=lambda r:
        r["total_gaze_movement"],
        reverse=True
    )[
        :WINDOWS_TO_SELECT
    ]

    selected["high_x_variation"] = sorted(
        rows,
        key=lambda r:
        r["gaze_x_std"],
        reverse=True
    )[
        :WINDOWS_TO_SELECT
    ]

    selected["high_depth_variation"] = sorted(
        rows,
        key=lambda r:
        r["depth_std"],
        reverse=True
    )[
        :WINDOWS_TO_SELECT
    ]

    selected["high_direction_change"] = sorted(
        rows,
        key=lambda r:
        r["direction_change"],
        reverse=True
    )[
        :WINDOWS_TO_SELECT
    ]

    return selected


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    frame_ids,
    row,
    category,
    image_directory,
    output_directory,
    image_type
):

    window_index = row["window_index"]

    # --------------------------------------------------------
    # Verify exactly 16 frames
    # --------------------------------------------------------

    if len(frame_ids) != TEMPORAL_LENGTH:

        print(
            f"WARNING: Window {window_index} "
            f"contains {len(frame_ids)} frames."
        )

        return False

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    images = []

    missing = []

    for frame_id in frame_ids:

        path = find_image(
            image_directory,
            frame_id
        )

        if path is None:

            missing.append(
                frame_id
            )

            continue

        try:

            image = Image.open(
                path
            ).convert("RGB")

            image.load()

            images.append(
                (frame_id, image)
            )

        except Exception as error:

            print(
                f"ERROR loading frame "
                f"{frame_id}: {error}"
            )

    if missing:

        print(
            f"WARNING: Missing {image_type} "
            f"frames: {missing}"
        )

    if not images:

        print(
            f"ERROR: No {image_type} images "
            f"could be loaded for window "
            f"{window_index}."
        )

        return False

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized_images = []

    for frame_id, image in images:

        image = image.resize(
            (
                THUMBNAIL_WIDTH,
                THUMBNAIL_HEIGHT
            ),
            Image.Resampling.LANCZOS
        )

        resized_images.append(
            (frame_id, image)
        )

    # --------------------------------------------------------
    # Create sheet
    # --------------------------------------------------------

    rows_needed = int(
        np.ceil(
            len(resized_images)
            / COLUMNS
        )
    )

    header_height = 100

    sheet_width = (
        COLUMNS *
        THUMBNAIL_WIDTH
    )

    sheet_height = (
        header_height
        +
        rows_needed *
        (
            THUMBNAIL_HEIGHT
            +
            LABEL_HEIGHT
        )
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        "white"
    )

    draw = ImageDraw.Draw(
        sheet
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header = (
        f"{category.upper()} | "
        f"{image_type.upper()} | "
        f"Window {window_index}\n"
        f"Frames {row['start_frame']} -> "
        f"{row['end_frame']}\n"
        f"Movement: "
        f"{row['total_gaze_movement']:.1f} | "
        f"X std: {row['gaze_x_std']:.1f} | "
        f"Y std: {row['gaze_y_std']:.1f}"
    )

    draw.text(
        (10, 10),
        header,
        fill="black"
    )

    # --------------------------------------------------------
    # Paste images
    # --------------------------------------------------------

    for index, (frame_id, image) in enumerate(
        resized_images
    ):

        column = index % COLUMNS

        row_number = index // COLUMNS

        x = (
            column *
            THUMBNAIL_WIDTH
        )

        y = (
            header_height
            +
            row_number *
            (
                THUMBNAIL_HEIGHT
                +
                LABEL_HEIGHT
            )
        )

        sheet.paste(
            image,
            (x, y)
        )

        draw.text(
            (
                x + 8,
                y + THUMBNAIL_HEIGHT + 7
            ),
            f"Frame {frame_id}",
            fill="black"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    filename = (
        f"{category}_"
        f"{image_type}_"
        f"window_{window_index}_"
        f"{row['start_frame']}_"
        f"{row['end_frame']}.jpg"
    )

    output_path = os.path.join(
        output_directory,
        filename
    )

    sheet.save(
        output_path,
        quality=95
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    saved = Image.open(
        output_path
    )

    saved.load()

    extrema = saved.getextrema()

    print(
        f"Created: {output_path}"
    )

    print(
        f"  Loaded images: {len(images)}/16"
    )

    print(
        f"  Size: {saved.size}"
    )

    print(
        f"  Pixel range: {extrema}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Corrected Temporal Behavior Inspection")
    print("=" * 70)

    # --------------------------------------------------------
    # Check feature file
    # --------------------------------------------------------

    if not os.path.exists(
        FEATURE_FILE
    ):

        print(
            "ERROR:",
            FEATURE_FILE,
            "not found."
        )

        print(
            "Run scan_temporal_windows.py first."
        )

        return

    # --------------------------------------------------------
    # Synchronization
    # --------------------------------------------------------

    frame_ids = get_synchronized_frame_ids()

    print()
    print(
        "Synchronized frame count:",
        len(frame_ids)
    )

    if len(frame_ids) < TEMPORAL_LENGTH:

        print(
            "ERROR: Not enough frames."
        )

        return

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = load_features()

    print(
        "Temporal windows:",
        len(features)
    )

    # --------------------------------------------------------
    # Verify expected number
    # --------------------------------------------------------

    expected_windows = (
        len(frame_ids)
        -
        TEMPORAL_LENGTH
        +
        1
    )

    print(
        "Expected windows:",
        expected_windows
    )

    if len(features) != expected_windows:

        print(
            "WARNING: Feature CSV window count "
            "does not match synchronized frame count."
        )

    # --------------------------------------------------------
    # Select candidates
    # --------------------------------------------------------

    selected = select_windows(
        features
    )

    # --------------------------------------------------------
    # Recreate output directory
    # --------------------------------------------------------

    if os.path.exists(
        OUTPUT_DIR
    ):

        shutil.rmtree(
            OUTPUT_DIR
        )

    os.makedirs(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Generate sheets
    # --------------------------------------------------------

    successful = 0

    for category, candidate_rows in selected.items():

        print()
        print(
            "-" * 70
        )

        print(
            category.upper()
        )

        print(
            "-" * 70
        )

        category_directory = os.path.join(
            OUTPUT_DIR,
            category
        )

        os.makedirs(
            category_directory,
            exist_ok=True
        )

        for row in candidate_rows:

            window_index = row[
                "window_index"
            ]

            # ------------------------------------------------
            # IMPORTANT:
            # Use the synchronized frame list.
            # Do NOT reconstruct it from filenames.
            # ------------------------------------------------

            start = window_index

            end = (
                window_index
                +
                TEMPORAL_LENGTH
            )

            window_frame_ids = frame_ids[
                start:end
            ]

            print()
            print(
                f"Window {window_index}: "
                f"{window_frame_ids[0]} -> "
                f"{window_frame_ids[-1]}"
            )

            # ------------------------------------------------
            # Driver sheet
            # ------------------------------------------------

            driver_ok = create_contact_sheet(
                window_frame_ids,
                row,
                category,
                FACE_DIR,
                category_directory,
                "driver"
            )

            # ------------------------------------------------
            # Scene sheet
            # ------------------------------------------------

            scene_ok = create_contact_sheet(
                window_frame_ids,
                row,
                category,
                SCENE_DIR,
                category_directory,
                "scene"
            )

            if driver_ok and scene_ok:

                successful += 1

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "INSPECTION COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        "Successful windows:",
        successful
    )

    print()
    print(
        "Output folder:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )


if __name__ == "__main__":
    main()