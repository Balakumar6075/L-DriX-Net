import os
import cv2

from backend.detection.face_detector import FaceDetector
from backend.detection.eye_detection import EyeDetector
from backend.detection.head_pose import HeadPoseEstimator
from backend.detection.gaze_estimation import GazeEstimator
from backend.detection.driver_state import DriverStateDetector


DATASET_PATH = r"dataset\Subject01_1_data"

FACE_DIR = os.path.join(
    DATASET_PATH,
    "face_ims"
)

OUTPUT_DIR = "state_candidate_images"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# Candidate regions discovered from the full scan.
CANDIDATE_FRAMES = [
    # Previously confirmed distraction candidates
    925, 928, 943, 946, 949,
    1610, 1613, 1616, 1619, 1622, 1625, 1628,

    # Current DROWSY candidates
    4670, 4676
]


face_detector = FaceDetector()
eye_detector = EyeDetector()
head_pose_estimator = HeadPoseEstimator()
gaze_estimator = GazeEstimator(
    scene_width=942,
    scene_height=489
)


for frame_id in CANDIDATE_FRAMES:

    filename = (
        f"{frame_id:08d}_face.png"
    )

    image_path = os.path.join(
        FACE_DIR,
        filename
    )

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        print(
            f"Frame {frame_id}: "
            "image not found"
        )

        continue


    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    face = face_detector.detect_largest(
        frame
    )

    if face is None:

        print(
            f"Frame {frame_id}: "
            "face not detected"
        )

        continue


    # --------------------------------------------------------
    # EYES
    # --------------------------------------------------------

    eye_features = eye_detector.extract(
        face
    )


    # --------------------------------------------------------
    # HEAD POSE
    # --------------------------------------------------------

    image_height, image_width = frame.shape[:2]

    head_pose = head_pose_estimator.estimate(
        face,
        image_width,
        image_height
    )


    # --------------------------------------------------------
    # GAZE
    # --------------------------------------------------------

    gaze = gaze_estimator.estimate(
        eye_features,
        head_pose
    )


    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    output = frame.copy()

    # Face rectangle
    cv2.rectangle(
        output,
        (
            int(face["x"]),
            int(face["y"])
        ),
        (
            int(face["x"] + face["w"]),
            int(face["y"] + face["h"])
        ),
        (0, 255, 0),
        2
    )


    # Eye features
    output = eye_detector.draw_features(
        output,
        eye_features
    )


    # Head pose information
    output = head_pose_estimator.draw_info(
        output,
        head_pose,
        position=(20, 30)
    )


    # --------------------------------------------------------
    # GAZE POINT
    # --------------------------------------------------------

    gaze_location = gaze.get(
        "gaze_location_2d",
        None
    )

    if gaze_location is not None:

        gx = int(
            round(gaze_location[0])
        )

        gy = int(
            round(gaze_location[1])
        )

        # Gaze coordinates belong to the scene
        # coordinate system, so we display them as text.
        cv2.putText(
            output,
            f"Gaze: ({gx}, {gy})",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # --------------------------------------------------------
    # FRAME NUMBER
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"Frame: {frame_id}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        f"{frame_id:08d}_candidate.jpg"
    )

    cv2.imwrite(
        output_path,
        output
    )


    print(
        f"Saved: {output_path}"
    )


print()
print("=" * 70)
print("CANDIDATE INSPECTION COMPLETE")
print("=" * 70)

print()
print(
    "Images saved to:"
)

print(
    OUTPUT_DIR
)