import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# ==============================================================
# PROJECT PATHS
# ==============================================================

DETECTION_DIR = Path(__file__).resolve().parent
BACKEND_DIR = DETECTION_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent


# ==============================================================
# IMPORT PATH FIX
# ==============================================================

# Make backend come before the project root.
#
# This prevents Python from accidentally importing:
#
#     project/detection/
#
# instead of:
#
#     project/backend/detection/

for path in [
    str(DETECTION_DIR),
    str(BACKEND_DIR),
    str(PROJECT_DIR),
]:
    while path in sys.path:
        sys.path.remove(path)

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(1, str(PROJECT_DIR))


# ==============================================================
# DETECTION MODULES
# ==============================================================

from detection.face_detector import FaceDetector
from detection.eye_detection import EyeDetector
from detection.head_pose import HeadPoseEstimator
from detection.gaze_estimation import GazeEstimator


# ==============================================================
# L-DRIX-NET
# ==============================================================

from inference import LDriXNetInference


# ==============================================================
# DATASET TEST IMAGES
# ==============================================================

DRIVER_IMAGE_PATH = (
    PROJECT_DIR
    / "dataset"
    / "Subject01_1_data"
    / "face_ims"
    / "00000193_face.png"
)

SCENE_IMAGE_PATH = (
    PROJECT_DIR
    / "dataset"
    / "Subject01_1_data"
    / "scene_ims"
    / "00000193_scene.png"
)


# ==============================================================
# HELPER
# ==============================================================

def cv2_to_pil(image):
    """
    Convert OpenCV BGR image to PIL RGB image.
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(rgb)


# ==============================================================
# MAIN
# ==============================================================

def main():

    print()
    print("=" * 70)
    print("L-DriX-Net LIVE INFERENCE INTEGRATION TEST")
    print("=" * 70)

    print()
    print("Project root:")
    print(f"  {PROJECT_DIR}")

    print()
    print("Backend:")
    print(f"  {BACKEND_DIR}")

    # ==========================================================
    # 0. VERIFY MODULE IMPORTS
    # ==========================================================

    print()
    print("[0] VERIFYING DETECTION MODULES")

    import detection.face_detector as face_detector_module
    import detection.eye_detection as eye_detection_module
    import detection.head_pose as head_pose_module
    import detection.gaze_estimation as gaze_estimation_module

    print()
    print("  Face detector:")
    print(
        f"    {face_detector_module.__file__}"
    )

    print()
    print("  Eye detector:")
    print(
        f"    {eye_detection_module.__file__}"
    )

    print()
    print("  Head pose:")
    print(
        f"    {head_pose_module.__file__}"
    )

    print()
    print("  Gaze estimator:")
    print(
        f"    {gaze_estimation_module.__file__}"
    )

    expected_detection_dir = (
        DETECTION_DIR.resolve()
    )

    actual_modules = [
        (
            "face_detector",
            Path(
                face_detector_module.__file__
            ).resolve(),
        ),
        (
            "eye_detection",
            Path(
                eye_detection_module.__file__
            ).resolve(),
        ),
        (
            "head_pose",
            Path(
                head_pose_module.__file__
            ).resolve(),
        ),
        (
            "gaze_estimation",
            Path(
                gaze_estimation_module.__file__
            ).resolve(),
        ),
    ]

    for name, path in actual_modules:

        if path.parent != expected_detection_dir:

            raise RuntimeError(
                f"Wrong {name} module loaded:\n"
                f"{path}\n\n"
                f"Expected module inside:\n"
                f"{expected_detection_dir}"
            )

    print()
    print(
        "  Detection imports: PASS"
    )

    # ==========================================================
    # 1. CHECK INPUT FILES
    # ==========================================================

    print()
    print("[1] CHECKING INPUT FILES")

    print()
    print("  Driver image:")
    print(
        f"    {DRIVER_IMAGE_PATH}"
    )

    if not DRIVER_IMAGE_PATH.exists():

        raise FileNotFoundError(
            "Driver image does not exist:\n"
            f"{DRIVER_IMAGE_PATH}"
        )

    print()
    print("  Scene image:")
    print(
        f"    {SCENE_IMAGE_PATH}"
    )

    if not SCENE_IMAGE_PATH.exists():

        raise FileNotFoundError(
            "Scene image does not exist:\n"
            f"{SCENE_IMAGE_PATH}"
        )

    print()
    print(
        "  Input files: PASS"
    )

    # ==========================================================
    # 2. LOAD DRIVER IMAGE
    # ==========================================================

    print()
    print("[2] LOADING DRIVER IMAGE")

    driver_cv = cv2.imread(
        str(DRIVER_IMAGE_PATH)
    )

    if driver_cv is None:

        raise RuntimeError(
            "Unable to load driver image."
        )

    driver_height, driver_width = (
        driver_cv.shape[:2]
    )

    print(
        f"  Width:  {driver_width}"
    )

    print(
        f"  Height: {driver_height}"
    )

    print(
        "  Driver image: PASS"
    )

    # ==========================================================
    # 3. LOAD SCENE IMAGE
    # ==========================================================

    print()
    print("[3] LOADING SCENE IMAGE")

    scene_cv = cv2.imread(
        str(SCENE_IMAGE_PATH)
    )

    if scene_cv is None:

        raise RuntimeError(
            "Unable to load scene image."
        )

    scene_height, scene_width = (
        scene_cv.shape[:2]
    )

    print(
        f"  Width:  {scene_width}"
    )

    print(
        f"  Height: {scene_height}"
    )

    print(
        "  Scene image: PASS"
    )

    # ==========================================================
    # 4. FACE DETECTION
    # ==========================================================

    print()
    print("[4] FACE DETECTION")

    face_detector = FaceDetector()

    face = face_detector.detect_largest(
        driver_cv
    )

    if face is None:

        raise RuntimeError(
            "No face detected."
        )

    print()
    print(
        "  Face detected:"
    )

    print(
        f"    x = {face['x']}"
    )

    print(
        f"    y = {face['y']}"
    )

    print(
        f"    w = {face['w']}"
    )

    print(
        f"    h = {face['h']}"
    )

    print(
        f"    confidence = "
        f"{face['confidence']:.4f}"
    )

    print()
    print(
        "  Face detection: PASS"
    )

    # ==========================================================
    # 5. EYE FEATURE EXTRACTION
    # ==========================================================

    print()
    print("[5] EYE FEATURE EXTRACTION")

    eye_detector = EyeDetector()

    eye_features = (
        eye_detector.extract(
            face
        )
    )

    if eye_features is None:

        raise RuntimeError(
            "Eye feature extraction failed."
        )

    print()
    print(
        "  Right eye:",
        eye_features[
            "right_eye"
        ],
    )

    print(
        "  Left eye:",
        eye_features[
            "left_eye"
        ],
    )

    print(
        "  Eye center:",
        eye_features[
            "eye_center"
        ],
    )

    print(
        "  Eye distance:",
        eye_features[
            "eye_distance"
        ],
    )

    print(
        "  Normalized eye distance:",
        eye_features[
            "eye_distance_normalized"
        ],
    )

    print()
    print(
        "  Eye feature extraction: PASS"
    )

    # ==========================================================
    # 6. HEAD POSE
    # ==========================================================

    print()
    print("[6] HEAD POSE ESTIMATION")

    head_pose_estimator = (
        HeadPoseEstimator()
    )

    head_pose = (
        head_pose_estimator.estimate(
            face,
            driver_width,
            driver_height,
        )
    )

    if head_pose is None:

        raise RuntimeError(
            "Head pose estimation failed."
        )

    print()
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

    print()
    print(
        "  Head pose estimation: PASS"
    )

    # ==========================================================
    # 7. GAZE ESTIMATION
    # ==========================================================

    print()
    print("[7] GAZE ESTIMATION")

    gaze_estimator = GazeEstimator(
        scene_width=scene_width,
        scene_height=scene_height,
    )

    gaze_result = (
        gaze_estimator.estimate(
            eye_features,
            head_pose,
        )
    )

    if gaze_result is None:

        raise RuntimeError(
            "Gaze estimation failed."
        )

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

    print()
    print(
        "  Gaze estimation: PASS"
    )

    # ==========================================================
    # 8. BUILD 24-D VECTOR
    # ==========================================================

    print()
    print("[8] BUILDING 24-D GAZE VECTOR")

    gaze_vector = np.asarray(
        gaze_result[
            "vector_24d"
        ],
        dtype=np.float32,
    )

    print()
    print(
        "  Vector shape:",
        gaze_vector.shape,
    )

    print(
        "  Vector length:",
        len(gaze_vector),
    )

    if gaze_vector.shape != (
        24,
    ):

        raise RuntimeError(
            "Invalid gaze vector shape: "
            f"{gaze_vector.shape}"
        )

    print()
    print(
        "  24-D vector:"
    )

    for index, value in enumerate(
        gaze_vector
    ):

        print(
            f"    [{index:02d}] "
            f"{value:.6f}"
        )

    # ==========================================================
    # 9. PYTORCH TENSOR
    # ==========================================================

    print()
    print("[9] CREATING PYTORCH INPUT")

    gaze_tensor = (
        torch.from_numpy(
            gaze_vector
        )
        .float()
        .unsqueeze(0)
    )

    print()
    print(
        "  Tensor shape:",
        tuple(
            gaze_tensor.shape
        ),
    )

    print(
        "  Tensor dtype:",
        gaze_tensor.dtype,
    )

    if tuple(
        gaze_tensor.shape
    ) != (
        1,
        24,
    ):

        raise RuntimeError(
            "Invalid gaze tensor shape: "
            f"{tuple(gaze_tensor.shape)}"
        )

    print()
    print(
        "  24-D PyTorch input: PASS"
    )

    # ==========================================================
    # 10. CONVERT IMAGES TO PIL
    # ==========================================================

    print()
    print("[10] PREPARING MODEL IMAGES")

    driver_pil = cv2_to_pil(
        driver_cv
    )

    scene_pil = cv2_to_pil(
        scene_cv
    )

    print(
        "  Driver image:",
        type(driver_pil).__name__,
    )

    print(
        "  Scene image:",
        type(scene_pil).__name__,
    )

    print()
    print(
        "  Model image preparation: PASS"
    )

    # ==========================================================
    # 11. LOAD TRAINED MODEL
    # ==========================================================

    print()
    print("[11] LOADING TRAINED L-DRIX-NET")

    model_engine = (
        LDriXNetInference()
    )

    print()
    print(
        "  L-DriX-Net loaded successfully."
    )

    # ----------------------------------------------------------
    # Determine device safely.
    #
    # LDriXNetInference does not expose a public .device
    # attribute in the current implementation.
    # ----------------------------------------------------------

    model_device = None

    if hasattr(
        model_engine,
        "model",
    ):

        try:

            model_device = next(
                model_engine.model.parameters()
            ).device

        except StopIteration:

            model_device = None

    if model_device is None:

        if torch.cuda.is_available():

            model_device = torch.device(
                "cuda"
            )

        else:

            model_device = torch.device(
                "cpu"
            )

    print(
        "  Device:",
        model_device,
    )

    print()
    print(
        "  Model loading: PASS"
    )

    # ==========================================================
    # 12. MOVE GAZE INPUT TO MODEL DEVICE
    # ==========================================================

    print()
    print("[12] MOVING GAZE INPUT TO MODEL DEVICE")

    gaze_tensor = (
        gaze_tensor.to(
            model_device
        )
    )

    print(
        "  Gaze tensor device:",
        gaze_tensor.device,
    )

    print()
    print(
        "  Device transfer: PASS"
    )

    # ==========================================================
    # 13. RUN ACTUAL L-DRIX-NET
    # ==========================================================

    print()
    print("[13] RUNNING ACTUAL MODEL INFERENCE")

    print()
    print(
        "  Inputs:"
    )

    print(
        "    Driver image"
    )

    print(
        "    Scene image"
    )

    print(
        "    Live 24-D gaze vector"
    )

    print()

    with torch.no_grad():

        prediction, heatmap, attention = (
            model_engine.predict(
                driver_pil,
                scene_pil,
                gaze_tensor,
            )
        )

    print(
        "  Model inference completed."
    )

    print()
    print(
        "  Model inference: PASS"
    )

    # ==========================================================
    # 14. PREDICTION OUTPUT
    # ==========================================================

    print()
    print("[14] PREDICTION OUTPUT")

    print(
        "  Prediction shape:",
        tuple(
            prediction.shape
        ),
    )

    expected_prediction_shape = (
        1,
        24,
    )

    if tuple(
        prediction.shape
    ) != expected_prediction_shape:

        raise RuntimeError(
            "Unexpected prediction shape.\n"
            f"Received: "
            f"{tuple(prediction.shape)}\n"
            f"Expected: "
            f"{expected_prediction_shape}"
        )

    prediction_values = (
        prediction.detach()
        .cpu()
        .flatten()
        .tolist()
    )

    print()

    for index, value in enumerate(
        prediction_values
    ):

        print(
            f"    [{index:02d}] "
            f"{value:.6f}"
        )

    print()
    print(
        "  Prediction: PASS"
    )

    # ==========================================================
    # 15. HEATMAP OUTPUT
    # ==========================================================

    print()
    print("[15] ATTENTION HEATMAP")

    print(
        "  Heatmap shape:",
        tuple(
            heatmap.shape
        ),
    )

    expected_heatmap_shape = (
        1,
        1,
        224,
        224,
    )

    if tuple(
        heatmap.shape
    ) != expected_heatmap_shape:

        raise RuntimeError(
            "Unexpected heatmap shape.\n"
            f"Received: "
            f"{tuple(heatmap.shape)}\n"
            f"Expected: "
            f"{expected_heatmap_shape}"
        )

    heatmap_cpu = (
        heatmap.detach()
        .cpu()
    )

    heatmap_min = float(
        heatmap_cpu.min().item()
    )

    heatmap_max = float(
        heatmap_cpu.max().item()
    )

    heatmap_mean = float(
        heatmap_cpu.mean().item()
    )

    print()
    print(
        f"  Minimum: {heatmap_min:.6f}"
    )

    print(
        f"  Maximum: {heatmap_max:.6f}"
    )

    print(
        f"  Mean:    {heatmap_mean:.6f}"
    )

    print()
    print(
        "  Attention heatmap: PASS"
    )

    # ==========================================================
    # 16. TEMPORAL ATTENTION
    # ==========================================================

    print()
    print("[16] TEMPORAL ATTENTION")

    print(
        "  Attention shape:",
        tuple(
            attention.shape
        ),
    )

    attention_values = (
        attention.detach()
        .cpu()
        .flatten()
        .tolist()
    )

    print(
        "  Attention values:",
        attention_values,
    )

    print()
    print(
        "  Temporal attention: PASS"
    )

    # ==========================================================
    # 17. FINAL RESULT
    # ==========================================================

    print()
    print("=" * 70)
    print("LIVE INFERENCE INTEGRATION SUCCESSFUL")
    print("=" * 70)

    print()

    print(
        "Face detection          : PASS"
    )

    print(
        "Eye geometry            : PASS"
    )

    print(
        "Head pose               : PASS"
    )

    print(
        "Gaze estimation         : PASS"
    )

    print(
        "24-D vector             : PASS"
    )

    print(
        "PyTorch input           : PASS"
    )

    print(
        "L-DriX-Net model        : PASS"
    )

    print(
        "Model inference         : PASS"
    )

    print(
        "24-D prediction         : PASS"
    )

    print(
        "Attention heatmap       : PASS"
    )

    print(
        "Temporal attention      : PASS"
    )

    print()
    print(
        "The complete live inference "
        "pipeline is working."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Connect the pipeline to FastAPI."
    )

    print()


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()