import inspect

from backend.detection.eye_detection import EyeDetector
from backend.detection.head_pose import HeadPoseEstimator
from backend.detection.face_detector import FaceDetector
from backend.detection.gaze_estimation import GazeEstimator


def inspect_class(cls):

    print()
    print("=" * 70)
    print(cls.__name__)
    print("=" * 70)

    print()
    print("Constructor:")
    print(inspect.signature(cls))

    print()
    print("Methods:")

    for name in dir(cls):

        if name.startswith("_"):
            continue

        attribute = getattr(cls, name)

        if callable(attribute):

            try:
                signature = inspect.signature(
                    attribute
                )
            except Exception:
                signature = "(signature unavailable)"

            print(
                f"  {name}{signature}"
            )


def main():

    print("=" * 70)
    print("L-DriX-Net Detection API Inspection")
    print("=" * 70)

    inspect_class(FaceDetector)
    inspect_class(EyeDetector)
    inspect_class(HeadPoseEstimator)
    inspect_class(GazeEstimator)

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()