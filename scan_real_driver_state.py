import os
import re
import cv2
import numpy as np

from backend.detection.face_detector import FaceDetector
from backend.detection.eye_detection import EyeDetector
from backend.detection.head_pose import HeadPoseEstimator
from backend.detection.gaze_estimation import GazeEstimator
from backend.detection.driver_state import DriverStateDetector


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"dataset\Subject01_1_data"

FACE_DIR = os.path.join(
    DATASET_PATH,
    "face_ims"
)

WINDOW_SIZE = 16


# ============================================================
# FIND FRAME FILES
# ============================================================

def get_frame_files():

    files = []

    for filename in os.listdir(FACE_DIR):

        if not filename.endswith("_face.png"):
            continue

        match = re.match(
            r"(\d+)_face\.png",
            filename
        )

        if match is None:
            continue

        frame_id = int(
            match.group(1)
        )

        files.append(
            (frame_id, filename)
        )

    files.sort(
        key=lambda x: x[0]
    )

    return files


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("FULL REAL-DATA DRIVER STATE ANALYSIS")
print("=" * 70)

print()
print("Initializing detectors...")

face_detector = FaceDetector()
eye_detector = EyeDetector()
head_pose_estimator = HeadPoseEstimator()
gaze_estimator = GazeEstimator(
    scene_width=942,
    scene_height=489
)

state_detector = DriverStateDetector(
    window_size=WINDOW_SIZE
)

print("Detectors initialized.")


# ============================================================
# FRAME LIST
# ============================================================

frame_files = get_frame_files()

print()
print("Total frames:")
print(len(frame_files))


# ============================================================
# PROCESS
# ============================================================

results = []

successful = 0
failed = 0

for index, (frame_id, filename) in enumerate(
    frame_files,
    start=1
):

    image_path = os.path.join(
        FACE_DIR,
        filename
    )

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        failed += 1
        continue

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    face = face_detector.detect_largest(
        frame
    )

    if face is None:

        failed += 1
        continue

    # --------------------------------------------------------
    # EYES
    # --------------------------------------------------------

    eye_features = eye_detector.extract(
        face
    )

    if eye_features is None:

        failed += 1
        continue

    # --------------------------------------------------------
    # HEAD POSE
    # --------------------------------------------------------

    image_height, image_width = frame.shape[:2]

    head_pose = head_pose_estimator.estimate(
        face,
        image_width,
        image_height
    )

    if head_pose is None:

        failed += 1
        continue

    # --------------------------------------------------------
    # GAZE
    # --------------------------------------------------------

    gaze = gaze_estimator.estimate(
        eye_features,
        head_pose
    )

    if gaze is None:

        failed += 1
        continue

    # --------------------------------------------------------
    # TEMPORAL STATE
    # --------------------------------------------------------

    state_result = state_detector.update({

        "head_pose": head_pose,

        "gaze": gaze,

        "eye_features": eye_features,

    })

    successful += 1

    if (
        state_result["state"]
        == "CALIBRATING"
    ):
        continue

    features = state_result.get(
        "features",
        {}
    )

    results.append({

        "frame_id": frame_id,

        "state": state_result["state"],

        "confidence": state_result["confidence"],

        "yaw_mean": features.get(
            "yaw_mean",
            0
        ),

        "yaw_std": features.get(
            "yaw_std",
            0
        ),

        "yaw_range": features.get(
            "yaw_range",
            0
        ),

        "pitch_mean": features.get(
            "pitch_mean",
            0
        ),

        "pitch_std": features.get(
            "pitch_std",
            0
        ),

        "pitch_range": features.get(
            "pitch_range",
            0
        ),

        "gaze_x_std": features.get(
            "gaze_x_std",
            0
        ),

        "gaze_y_std": features.get(
            "gaze_y_std",
            0
        ),

        "gaze_x_range": features.get(
            "gaze_x_range",
            0
        ),

        "gaze_y_range": features.get(
            "gaze_y_range",
            0
        ),

        "gaze_movement": features.get(
            "gaze_movement",
            0
        ),

        "concentrated_score": state_result[
            "scores"
        ]["CONCENTRATED"],

        "distracted_score": state_result[
            "scores"
        ]["DISTRACTED"],

        "drowsy_score": state_result[
            "scores"
        ]["DROWSY"],

    })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 100 == 0:

        print(
            f"Processed "
            f"{index}/{len(frame_files)} "
            f"frames..."
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Successful frames:")
print(successful)

print()
print("Failed frames:")
print(failed)

print()
print("Temporal windows:")
print(len(results))


# ============================================================
# STATE COUNTS
# ============================================================

state_counts = {
    "CONCENTRATED": 0,
    "DISTRACTED": 0,
    "DROWSY": 0,
}

for result in results:

    state = result["state"]

    if state in state_counts:

        state_counts[state] += 1


print()
print("CURRENT STATE DISTRIBUTION")
print("-" * 70)

for state, count in state_counts.items():

    print(
        f"{state:15s}: {count}"
    )


# ============================================================
# EXTREME WINDOWS
# ============================================================

print()
print("=" * 70)
print("MOST EXTREME WINDOWS")
print("=" * 70)


# ------------------------------------------------------------
# Highest yaw
# ------------------------------------------------------------

highest_yaw = sorted(
    results,
    key=lambda x: abs(
        x["yaw_mean"]
    ),
    reverse=True
)[:10]

print()
print("TOP 10 HIGHEST HEAD ROTATION")
print("-" * 70)

for item in highest_yaw:

    print(
        f"Frame {item['frame_id']:5d} | "
        f"Yaw {item['yaw_mean']:7.2f}° | "
        f"Pitch {item['pitch_mean']:7.2f}° | "
        f"State {item['state']}"
    )


# ------------------------------------------------------------
# Highest pitch
# ------------------------------------------------------------

highest_pitch = sorted(
    results,
    key=lambda x: x["pitch_mean"],
    reverse=True
)[:10]

print()
print("TOP 10 HIGHEST DOWNWARD PITCH")
print("-" * 70)

for item in highest_pitch:

    print(
        f"Frame {item['frame_id']:5d} | "
        f"Pitch {item['pitch_mean']:7.2f}° | "
        f"Yaw {item['yaw_mean']:7.2f}° | "
        f"State {item['state']}"
    )


# ------------------------------------------------------------
# Highest gaze movement
# ------------------------------------------------------------

highest_gaze = sorted(
    results,
    key=lambda x: x["gaze_movement"],
    reverse=True
)[:10]

print()
print("TOP 10 HIGHEST GAZE MOVEMENT")
print("-" * 70)

for item in highest_gaze:

    print(
        f"Frame {item['frame_id']:5d} | "
        f"Gaze movement {item['gaze_movement']:8.2f} | "
        f"Yaw {item['yaw_mean']:7.2f}° | "
        f"Pitch {item['pitch_mean']:7.2f}° | "
        f"State {item['state']}"
    )


# ------------------------------------------------------------
# Lowest activity
# ------------------------------------------------------------

lowest_activity = sorted(
    results,
    key=lambda x: (
        x["gaze_x_std"]
        + x["gaze_y_std"]
    )
)[:10]

print()
print("TOP 10 LOWEST GAZE ACTIVITY")
print("-" * 70)

for item in lowest_activity:

    activity = (
        item["gaze_x_std"]
        + item["gaze_y_std"]
    )

    print(
        f"Frame {item['frame_id']:5d} | "
        f"Activity {activity:7.2f} | "
        f"Yaw {item['yaw_mean']:7.2f}° | "
        f"Pitch {item['pitch_mean']:7.2f}° | "
        f"State {item['state']}"
    )


# ============================================================
# SAVE CSV
# ============================================================

output_file = "real_driver_state_analysis.csv"

try:

    import csv

    with open(
        output_file,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print(
        "Saved analysis:"
    )

    print(
        output_file
    )

except Exception as error:

    print()
    print(
        "Could not save CSV:"
    )

    print(error)


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)