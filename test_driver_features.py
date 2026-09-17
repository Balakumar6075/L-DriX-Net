import os
import cv2
import numpy as np

from backend.detection.face_detector import FaceDetector
from backend.detection.eye_detection import EyeDetector
from backend.detection.head_pose import HeadPoseEstimator
from backend.detection.gaze_estimation import GazeEstimator


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"dataset\Subject01_1_data"

FACE_DIR = os.path.join(
    BASE_DIR,
    "face_ims"
)

# Use a known driver frame
FRAME_ID = 3104


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Driver Feature Extraction Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image_path = os.path.join(
        FACE_DIR,
        f"{FRAME_ID:08d}_face.png"
    )

    print()
    print("Image:")
    print(image_path)

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        print()
        print("ERROR: Could not load image.")

        return

    print(
        "Image shape:",
        frame.shape
    )

    # --------------------------------------------------------
    # Initialize detectors
    # --------------------------------------------------------

    print()
    print("Initializing detectors...")

    face_detector = FaceDetector()

    eye_detector = EyeDetector()

    head_pose_estimator = (
        HeadPoseEstimator()
    )

    gaze_estimator = GazeEstimator()

    print("Detectors initialized.")

    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("FACE DETECTION")
    print("-" * 70)

    face = face_detector.detect_largest(
        frame
    )

    print(
        "Face type:",
        type(face)
    )

    print(
        "Face:"
    )

    print(face)

    if face is None:

        print(
            "ERROR: No face detected."
        )

        return

    # --------------------------------------------------------
    # Crop face
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("FACE CROP")
    print("-" * 70)

    cropped_face = (
        face_detector.crop_face(
            frame,
            face
        )
    )

    print(
        "Cropped face type:",
        type(cropped_face)
    )

    print(
        "Cropped face shape:",
        cropped_face.shape
    )

    # --------------------------------------------------------
    # Eye features
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("EYE FEATURES")
    print("-" * 70)

    eye_features = eye_detector.extract(
        face
    )

    print(
        "Eye feature type:",
        type(eye_features)
    )

    print()
    print("Eye features:")

    print(eye_features)

    # --------------------------------------------------------
    # Head pose
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("HEAD POSE")
    print("-" * 70)

    image_height, image_width = (
        frame.shape[:2]
    )

    head_pose = (
        head_pose_estimator.estimate(
            face,
            image_width,
            image_height
        )
    )

    print(
        "Head pose type:",
        type(head_pose)
    )

    print()
    print("Head pose:")

    print(head_pose)

    # --------------------------------------------------------
    # Gaze
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("GAZE")
    print("-" * 70)

    try:

        gaze = gaze_estimator.estimate(
            eye_features,
            head_pose
        )

        print(
            "Gaze type:",
            type(gaze)
        )

        print()
        print("Gaze:")

        print(gaze)

    except Exception as error:

        print(
            "Gaze estimation error:"
        )

        print(
            repr(error)
        )

    # --------------------------------------------------------
    # Gaze vector
    # --------------------------------------------------------

    try:

        gaze_vector = (
            gaze_estimator.build_vector(
                eye_features,
                head_pose
            )
        )

        print()
        print(
            "Gaze vector type:",
            type(gaze_vector)
        )

        print()
        print("Gaze vector:")

        print(gaze_vector)

        if isinstance(
            gaze_vector,
            np.ndarray
        ):

            print()
            print(
                "Gaze vector shape:",
                gaze_vector.shape
            )

    except Exception as error:

        print(
            "Gaze vector error:"
        )

        print(
            repr(error)
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()