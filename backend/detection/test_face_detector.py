import cv2
import numpy as np

from face_detector import FaceDetector
from eye_detection import EyeDetector
from head_pose import HeadPoseEstimator
from gaze_estimation import GazeEstimator


# ==============================================================
# Input image
# ==============================================================

IMAGE_PATH = (
    r"C:\Users\Balakumar B\OneDrive\Desktop\L-DriX-Net"
    r"\dataset\Subject01_1_data"
    r"\face_ims\00000193_face.png"
)


# ==============================================================
# Output image
# ==============================================================

OUTPUT_PATH = (
    r"C:\Users\Balakumar B\OneDrive\Desktop\L-DriX-Net"
    r"\backend\detection"
    r"\gaze_estimation_test.png"
)


# ==============================================================
# Main
# ==============================================================

def main():

    print()
    print("=" * 60)
    print("L-DriX-Net Live Gaze Feature Test")
    print("=" * 60)

    # ==========================================================
    # 1. Load image
    # ==========================================================

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:

        raise RuntimeError(
            "Unable to read image:\n"
            + IMAGE_PATH
        )

    height, width = (
        image.shape[:2]
    )

    print()
    print("[IMAGE]")

    print(
        f"  Width:  {width}"
    )

    print(
        f"  Height: {height}"
    )

    # ==========================================================
    # 2. Face detection
    # ==========================================================

    print()
    print("[1] FACE DETECTION")

    face_detector = FaceDetector()

    face = face_detector.detect_largest(
        image
    )

    if face is None:

        print(
            "  No face detected."
        )

        return

    print(
        f"  x = {face['x']}"
    )

    print(
        f"  y = {face['y']}"
    )

    print(
        f"  w = {face['w']}"
    )

    print(
        f"  h = {face['h']}"
    )

    print(
        f"  confidence = "
        f"{face['confidence']:.4f}"
    )

    # ==========================================================
    # 3. Eye feature extraction
    # ==========================================================

    print()
    print("[2] EYE GEOMETRY")

    eye_detector = EyeDetector()

    eye_features = eye_detector.extract(
        face
    )

    if eye_features is None:

        print(
            "  Eye feature extraction failed."
        )

        return

    print(
        "  Right eye:",
        eye_features["right_eye"],
    )

    print(
        "  Left eye:",
        eye_features["left_eye"],
    )

    print(
        "  Eye center:",
        eye_features["eye_center"],
    )

    print(
        "  Eye distance:",
        f"{eye_features['eye_distance']:.4f}",
    )

    print(
        "  Eye center offset:",
        eye_features[
            "eye_center_offset"
        ],
    )

    print(
        "  Eye center offset normalized:",
        eye_features[
            "eye_center_offset_normalized"
        ],
    )

    # ==========================================================
    # 4. Head pose
    # ==========================================================

    print()
    print("[3] HEAD POSE")

    pose_estimator = (
        HeadPoseEstimator()
    )

    head_pose = pose_estimator.estimate(
        face,
        width,
        height,
    )

    if head_pose is None:

        print(
            "  Head pose estimation failed."
        )

        return

    print(
        f"  Yaw:   "
        f"{head_pose['yaw']:.2f} degrees"
    )

    print(
        f"  Pitch: "
        f"{head_pose['pitch']:.2f} degrees"
    )

    print(
        f"  Roll:  "
        f"{head_pose['roll']:.2f} degrees"
    )

    print(
        f"  Yaw normalized: "
        f"{head_pose['yaw_normalized']:.4f}"
    )

    print(
        f"  Pitch normalized: "
        f"{head_pose['pitch_normalized']:.4f}"
    )

    print(
        f"  Roll normalized: "
        f"{head_pose['roll_normalized']:.4f}"
    )

    # ==========================================================
    # 5. Gaze estimation
    # ==========================================================

    print()
    print("[4] GAZE ESTIMATION")

    gaze_estimator = GazeEstimator(
        scene_width=942,
        scene_height=489,
    )

    gaze_result = gaze_estimator.estimate(
        eye_features,
        head_pose,
    )

    if gaze_result is None:

        print(
            "  Gaze estimation failed."
        )

        return

    print()
    print(
        "  Gaze 2D location:"
    )

    print(
        "   ",
        gaze_result[
            "gaze_location_2d"
        ],
    )

    print()
    print(
        "  Gaze 3D location:"
    )

    print(
        "   ",
        gaze_result[
            "gaze_location_3d"
        ],
    )

    print()
    print(
        "  Left gaze direction:"
    )

    print(
        "   ",
        gaze_result[
            "left_gaze_direction"
        ],
    )

    print()
    print(
        "  Right gaze direction:"
    )

    print(
        "   ",
        gaze_result[
            "right_gaze_direction"
        ],
    )

    print()
    print(
        "  Average gaze direction:"
    )

    print(
        "   ",
        gaze_result[
            "average_gaze_direction"
        ],
    )

    # ==========================================================
    # 6. 24-D vector
    # ==========================================================

    print()
    print("[5] L-DRIX-NET 24-D INPUT")

    vector = np.asarray(
        gaze_result[
            "vector_24d"
        ],
        dtype=np.float32,
    )

    print()
    print(
        "  Shape:",
        vector.shape,
    )

    print(
        "  Number of dimensions:",
        len(vector),
    )

    if len(vector) != 24:

        raise RuntimeError(
            "ERROR: Gaze vector is not "
            "24-dimensional."
        )

    print()
    print(
        "  24-D vector:"
    )

    for index, value in enumerate(
        vector
    ):

        print(
            f"    [{index:02d}] "
            f"{value:.6f}"
        )

    # ==========================================================
    # 7. Create visualization
    # ==========================================================

    print()
    print("[6] VISUALIZATION")

    output = image.copy()

    # ----------------------------------------------------------
    # Face bounding box
    # ----------------------------------------------------------

    x = int(face["x"])
    y = int(face["y"])
    w = int(face["w"])
    h = int(face["h"])

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )

    # ----------------------------------------------------------
    # Face landmarks
    # ----------------------------------------------------------

    landmarks = [
        (
            face["right_eye"],
            (255, 0, 0),
        ),
        (
            face["left_eye"],
            (0, 0, 255),
        ),
        (
            face["nose"],
            (0, 255, 255),
        ),
        (
            face["right_mouth"],
            (255, 0, 255),
        ),
        (
            face["left_mouth"],
            (0, 255, 255),
        ),
    ]

    for point, color in landmarks:

        px = int(
            round(point[0])
        )

        py = int(
            round(point[1])
        )

        cv2.circle(
            output,
            (px, py),
            5,
            color,
            -1,
        )

    # ----------------------------------------------------------
    # Draw eye center
    # ----------------------------------------------------------

    eye_center = (
        int(
            round(
                eye_features[
                    "eye_center"
                ][0]
            )
        ),
        int(
            round(
                eye_features[
                    "eye_center"
                ][1]
            )
        ),
    )

    cv2.circle(
        output,
        eye_center,
        6,
        (255, 255, 255),
        -1,
    )

    # ----------------------------------------------------------
    # Draw head-pose visualization
    # ----------------------------------------------------------

    output = pose_estimator.draw_axes(
        output,
        face,
        head_pose,
    )

    # ----------------------------------------------------------
    # Draw head-pose information
    # ----------------------------------------------------------

    output = pose_estimator.draw_info(
        output,
        head_pose,
        position=(20, 35),
    )

    # ==========================================================
    # 8. Draw gaze direction
    # ==========================================================

    direction = np.asarray(
        gaze_result[
            "average_gaze_direction"
        ],
        dtype=np.float32,
    )

    nose_x = int(
        round(
            face["nose"][0]
        )
    )

    nose_y = int(
        round(
            face["nose"][1]
        )
    )

    # ----------------------------------------------------------
    # Only use X and Y for visualization.
    #
    # Z represents forward depth.
    # ----------------------------------------------------------

    arrow_length = max(
        80,
        int(
            face["w"] * 0.75
        ),
    )

    end_x = int(
        round(
            nose_x
            + direction[0]
            * arrow_length
        )
    )

    end_y = int(
        round(
            nose_y
            - direction[1]
            * arrow_length
        )
    )

    cv2.arrowedLine(
        output,
        (nose_x, nose_y),
        (end_x, end_y),
        (255, 255, 0),
        4,
        tipLength=0.15,
    )

    # ----------------------------------------------------------
    # Gaze label
    # ----------------------------------------------------------

    cv2.putText(
        output,
        "Estimated Gaze",
        (
            nose_x + 10,
            nose_y - 15,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # ==========================================================
    # 9. Save visualization
    # ==========================================================

    success = cv2.imwrite(
        OUTPUT_PATH,
        output,
    )

    if not success:

        raise RuntimeError(
            "Unable to save output image:\n"
            + OUTPUT_PATH
        )

    print()
    print(
        "  Output saved:"
    )

    print(
        "  " + OUTPUT_PATH
    )

    # ==========================================================
    # Final result
    # ==========================================================

    print()
    print("=" * 60)
    print("COMPLETE GAZE FEATURE PIPELINE SUCCESSFUL")
    print("=" * 60)
    print()

    print(
        "Face detection       : PASS"
    )

    print(
        "Eye geometry          : PASS"
    )

    print(
        "Head pose             : PASS"
    )

    print(
        "Gaze estimation       : PASS"
    )

    print(
        "24-D vector           : PASS"
    )

    print()
    print(
        "Ready for L-DriX-Net inference."
    )

    print()


# ==============================================================
# Entry point
# ==============================================================

if __name__ == "__main__":
    main()