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

MAX_FRAMES = 200


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
print("REAL DRIVER STATE TEST")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_PATH)

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
# GET FRAMES
# ============================================================

frame_files = get_frame_files()

print()
print("Total face frames found:")
print(len(frame_files))

frame_files = frame_files[:MAX_FRAMES]

print()
print("Frames used:")
print(len(frame_files))


# ============================================================
# PROCESS FRAMES
# ============================================================

results = []

successful_frames = 0

failed_frames = 0


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

        failed_frames += 1

        continue

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    face = face_detector.detect_largest(
        frame
    )

    if face is None:

        failed_frames += 1

        print(
            f"[{index:03d}] "
            f"Frame {frame_id}: "
            f"FACE NOT DETECTED"
        )

        continue

    # --------------------------------------------------------
    # EYE FEATURES
    # --------------------------------------------------------

    eye_features = eye_detector.extract(
        face
    )

    if eye_features is None:

        failed_frames += 1

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

        failed_frames += 1

        continue

    # --------------------------------------------------------
    # GAZE
    # --------------------------------------------------------

    gaze = gaze_estimator.estimate(
        eye_features,
        head_pose
    )

    if gaze is None:

        failed_frames += 1

        continue

    # --------------------------------------------------------
    # TEMPORAL STATE
    # --------------------------------------------------------

    frame_data = {

        "head_pose": head_pose,

        "gaze": gaze,

        "eye_features": eye_features,

    }

    state_result = state_detector.update(
        frame_data
    )

    successful_frames += 1

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    if (
        state_result["state"]
        != "CALIBRATING"
    ):

        results.append(
            state_result
        )

        print(
            f"[{index:03d}] "
            f"Frame {frame_id:5d} | "
            f"State: "
            f"{state_result['state']:12s} | "
            f"Confidence: "
            f"{state_result['confidence']:.2f}"
        )

    else:

        print(
            f"[{index:03d}] "
            f"Frame {frame_id:5d} | "
            f"CALIBRATING "
            f"({state_result['frames']}/"
            f"{state_result['required_frames']})"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("REAL DATA TEST SUMMARY")
print("=" * 70)

print()
print("Successful frames:")
print(successful_frames)

print()
print("Failed frames:")
print(failed_frames)

print()
print("Completed temporal windows:")
print(len(results))


# ============================================================
# COUNT STATES
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
print("STATE COUNTS")
print("-" * 70)

for state, count in state_counts.items():

    print(
        f"{state:15s}: {count}"
    )


# ============================================================
# PERCENTAGES
# ============================================================

total_states = sum(
    state_counts.values()
)

print()
print("STATE PERCENTAGES")
print("-" * 70)

if total_states > 0:

    for state, count in state_counts.items():

        percentage = (
            count
            / total_states
            * 100
        )

        print(
            f"{state:15s}: "
            f"{percentage:.2f}%"
        )

else:

    print("No completed windows.")


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)