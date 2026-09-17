import base64
import io
import os
import sys

import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


# ============================================================
# PATH SETUP
# ============================================================

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BACKEND_DIR
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from backend.inference import LDriXNetInference

from backend.detection.face_detector import (
    FaceDetector
)

from backend.detection.eye_detection import (
    EyeDetector
)

from backend.detection.head_pose import (
    HeadPoseEstimator
)

from backend.detection.gaze_estimation import (
    GazeEstimator
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="L-DriX-Net API",
    version="1.0.0",
    description=(
        "Backend API for L-DriX-Net driver "
        "gaze and attention estimation."
    )
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# GLOBAL COMPONENTS
# ============================================================

model_engine = None

face_detector = None

eye_detector = None

head_pose_estimator = None

gaze_estimator = None


# ============================================================
# HEATMAP → BASE64 PNG
# ============================================================

def heatmap_to_base64(
    heatmap_tensor
):
    """
    Convert the actual L-DriX-Net heatmap tensor
    into a colorized PNG encoded as Base64.

    Expected tensor:

        [1, 1, 224, 224]
    """

    heatmap = (
        heatmap_tensor
        .detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    minimum = float(
        np.min(heatmap)
    )

    maximum = float(
        np.max(heatmap)
    )

    # --------------------------------------------------------
    # Normalize to 0–255
    # --------------------------------------------------------

    if maximum - minimum < 1e-8:

        normalized = np.zeros_like(
            heatmap,
            dtype=np.uint8
        )

    else:

        normalized = (
            (
                heatmap - minimum
            )
            /
            (
                maximum - minimum
            )
            *
            255.0
        )

        normalized = (
            normalized
            .clip(0, 255)
            .astype(np.uint8)
        )

    # --------------------------------------------------------
    # Apply color map
    # --------------------------------------------------------

    colored = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    # OpenCV BGR → RGB
    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        colored
    )

    # --------------------------------------------------------
    # Encode PNG
    # --------------------------------------------------------

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return encoded


# ============================================================
# HEATMAP → SCENE SIZE
# ============================================================

def heatmap_to_scene_base64(
    heatmap_tensor,
    scene_width,
    scene_height
):
    """
    Resize the 224×224 L-DriX-Net heatmap to the
    actual scene-image dimensions.
    """

    heatmap = (
        heatmap_tensor
        .detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    minimum = float(
        np.min(heatmap)
    )

    maximum = float(
        np.max(heatmap)
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    if maximum - minimum < 1e-8:

        normalized = np.zeros_like(
            heatmap,
            dtype=np.uint8
        )

    else:

        normalized = (
            (
                heatmap - minimum
            )
            /
            (
                maximum - minimum
            )
            *
            255.0
        )

        normalized = (
            normalized
            .clip(0, 255)
            .astype(np.uint8)
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv2.resize(
        normalized,
        (
            int(scene_width),
            int(scene_height)
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Color map
    # --------------------------------------------------------

    colored = cv2.applyColorMap(
        resized,
        cv2.COLORMAP_JET
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        colored
    )

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return encoded


# ============================================================
# READ UPLOADED IMAGE
# ============================================================

async def read_uploaded_image(
    upload_file: UploadFile
):

    try:

        contents = await upload_file.read()

        if not contents:

            raise ValueError(
                "Uploaded image is empty."
            )

        # ----------------------------------------------------
        # PIL
        # ----------------------------------------------------

        pil_image = Image.open(
            io.BytesIO(contents)
        ).convert(
            "RGB"
        )

        # ----------------------------------------------------
        # OpenCV
        # ----------------------------------------------------

        cv_image = np.array(
            pil_image
        )

        cv_image = cv2.cvtColor(
            cv_image,
            cv2.COLOR_RGB2BGR
        )

        return (
            pil_image,
            cv_image
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid uploaded image: "
                f"{error}"
            )
        )


# ============================================================
# FACE BOUNDING BOX
# ============================================================

def get_face_bbox(
    face
):
    """
    FaceDetector returns a dictionary like:

    {
        "x": ...,
        "y": ...,
        "w": ...,
        "h": ...,
        "confidence": ...,
        "landmarks": ...
    }
    """

    if not isinstance(
        face,
        dict
    ):

        raise ValueError(
            "Unexpected face detector output. "
            f"Received: {type(face)}"
        )

    return (
        int(face["x"]),
        int(face["y"]),
        int(face["w"]),
        int(face["h"])
    )


# ============================================================
# CROP FACE
# ============================================================

def crop_face_from_dict(
    image,
    face,
    padding=0.2
):
    """
    Crop detected face from an OpenCV image.
    """

    x, y, w, h = get_face_bbox(
        face
    )

    image_height, image_width = (
        image.shape[:2]
    )

    pad_x = int(
        w * padding
    )

    pad_y = int(
        h * padding
    )

    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        image_width,
        x + w + pad_x
    )

    y2 = min(
        image_height,
        y + h + pad_y
    )

    cropped = image[
        y1:y2,
        x1:x2
    ]

    return cropped


# ============================================================
# SAFE FLOAT CONVERSION
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        return float(
            value
        )

    except Exception:

        return float(
            default
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global model_engine
    global face_detector
    global eye_detector
    global head_pose_estimator
    global gaze_estimator

    print()
    print("=" * 70)
    print("Starting L-DriX-Net API")
    print("=" * 70)

    # ========================================================
    # 1. MODEL
    # ========================================================

    print()
    print(
        "[1] Loading trained L-DriX-Net model..."
    )

    model_engine = LDriXNetInference()

    device = str(
        next(
            model_engine.model.parameters()
        ).device
    )

    print(
        "    L-DriX-Net loaded successfully."
    )

    print(
        f"    Device: {device}"
    )

    # ========================================================
    # 2. FACE DETECTOR
    # ========================================================

    print()
    print(
        "[2] Initializing face detector..."
    )

    face_detector = FaceDetector()

    print(
        "    Face detector initialized."
    )

    # ========================================================
    # 3. EYE DETECTOR
    # ========================================================

    print()
    print(
        "[3] Initializing eye detector..."
    )

    eye_detector = EyeDetector()

    print(
        "    Eye detector initialized."
    )

    # ========================================================
    # 4. HEAD POSE
    # ========================================================

    print()
    print(
        "[4] Initializing head pose estimator..."
    )

    head_pose_estimator = (
        HeadPoseEstimator()
    )

    print(
        "    Head pose estimator initialized."
    )

    # ========================================================
    # 5. GAZE ESTIMATOR
    # ========================================================

    print()
    print(
        "[5] Initializing gaze estimator..."
    )

    gaze_estimator = GazeEstimator()

    print(
        "    Gaze estimator initialized."
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "L-DriX-Net backend initialization complete"
    )
    print("=" * 70)
    print()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name":
            "L-DriX-Net API",

        "version":
            "1.0.0",

        "status":
            "online",

        "docs":
            "/docs"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    model_loaded = (
        model_engine is not None
    )

    face_loaded = (
        face_detector is not None
    )

    eye_loaded = (
        eye_detector is not None
    )

    head_pose_loaded = (
        head_pose_estimator is not None
    )

    gaze_loaded = (
        gaze_estimator is not None
    )

    device = "unknown"

    if model_loaded:

        device = str(
            next(
                model_engine.model.parameters()
            ).device
        )

    return {

        "status":
            "online",

        "model":
            "L-DriX-Net",

        "device":
            device,

        "model_loaded":
            model_loaded,

        "face_detector_loaded":
            face_loaded,

        "eye_detector_loaded":
            eye_loaded,

        "head_pose_loaded":
            head_pose_loaded,

        "gaze_estimator_loaded":
            gaze_loaded
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    if model_engine is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    device = str(
        next(
            model_engine.model.parameters()
        ).device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model_engine.model.parameters()
    )

    return {

        "name":
            "L-DriX-Net",

        "parameters":
            parameter_count,

        "input_size": [
            224,
            224
        ],

        "gaze_dimension":
            24,

        "prediction_dimension":
            24,

        "heatmap_size": [
            224,
            224
        ],

        "device":
            device
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(

    driver_image: UploadFile = File(...),

    scene_image: UploadFile = File(...)

):

    print()
    print("=" * 70)
    print("NEW L-DRIX-NET PREDICTION REQUEST")
    print("=" * 70)

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if model_engine is None:

        raise HTTPException(
            status_code=503,
            detail="L-DriX-Net model is not loaded."
        )

    # ========================================================
    # READ DRIVER IMAGE
    # ========================================================

    print(
        "[1] Reading driver image..."
    )

    driver_pil, driver_cv = (
        await read_uploaded_image(
            driver_image
        )
    )

    print(
        f"    Driver image: "
        f"{driver_cv.shape[1]}x"
        f"{driver_cv.shape[0]}"
    )

    # ========================================================
    # READ SCENE IMAGE
    # ========================================================

    print(
        "[2] Reading scene image..."
    )

    scene_pil, scene_cv = (
        await read_uploaded_image(
            scene_image
        )
    )

    print(
        f"    Scene image: "
        f"{scene_cv.shape[1]}x"
        f"{scene_cv.shape[0]}"
    )

    # ========================================================
    # IMAGE DIMENSIONS
    # ========================================================

    driver_height, driver_width = (
        driver_cv.shape[:2]
    )

    scene_height, scene_width = (
        scene_cv.shape[:2]
    )

    # ========================================================
    # FACE DETECTION
    # ========================================================

    print(
        "[3] Detecting driver face..."
    )

    face = face_detector.detect_largest(
        driver_cv
    )

    if face is None:

        raise HTTPException(
            status_code=422,
            detail=(
                "No face detected in "
                "driver image."
            )
        )

    print(
        "    Face detected."
    )

    print(
        f"    x={face['x']}, "
        f"y={face['y']}, "
        f"w={face['w']}, "
        f"h={face['h']}"
    )

    print(
        f"    confidence="
        f"{face['confidence']:.4f}"
    )

    # ========================================================
    # FACE CROP
    # ========================================================

    print(
        "[4] Cropping driver face..."
    )

    face_crop = crop_face_from_dict(
        driver_cv,
        face,
        padding=0.2
    )

    if (
        face_crop is None
        or face_crop.size == 0
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                "Unable to crop detected face."
            )
        )

    face_crop_rgb = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )

    face_pil = Image.fromarray(
        face_crop_rgb
    )

    print(
        f"    Face crop: "
        f"{face_crop.shape[1]}x"
        f"{face_crop.shape[0]}"
    )

    # ========================================================
    # EYE FEATURES
    # ========================================================

    print(
        "[5] Extracting eye features..."
    )

    eye_features = eye_detector.extract(
        face
    )

    print(
        "    Eye features extracted."
    )

    # ========================================================
    # HEAD POSE
    # ========================================================

    print(
        "[6] Estimating head pose..."
    )

    head_pose = (
        head_pose_estimator.estimate(
            face,
            driver_width,
            driver_height
        )
    )

    print(
        f"    Yaw: "
        f"{safe_float(head_pose.get('yaw')):.2f}°"
    )

    print(
        f"    Pitch: "
        f"{safe_float(head_pose.get('pitch')):.2f}°"
    )

    print(
        f"    Roll: "
        f"{safe_float(head_pose.get('roll')):.2f}°"
    )

    # ========================================================
    # GAZE ESTIMATION
    # ========================================================

    print(
        "[7] Estimating gaze..."
    )

    gaze = gaze_estimator.estimate(
        eye_features,
        head_pose
    )

    print(
        "    Gaze estimated."
    )

    # ========================================================
    # 24-D GAZE VECTOR
    # ========================================================

    print(
        "[8] Building 24-D gaze vector..."
    )

    gaze_vector = (
        gaze_estimator.build_vector(
            eye_features,
            head_pose
        )
    )

    if len(gaze_vector) != 24:

        raise HTTPException(
            status_code=500,
            detail=(
                "Gaze estimator returned "
                f"{len(gaze_vector)} values. "
                "Expected 24."
            )
        )

    print(
        "    24-D gaze vector created."
    )

    # ========================================================
    # ACTUAL L-DRIX-NET MODEL
    # ========================================================

    print(
        "[9] Running L-DriX-Net..."
    )

    prediction, heatmap, attention = (
        model_engine.predict(
            face_pil,
            scene_pil,
            gaze_vector
        )
    )

    print(
        "    Model inference completed."
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction_cpu = (
        prediction
        .detach()
        .cpu()
    )

    prediction_values = (
        prediction_cpu
        .flatten()
        .tolist()
    )

    print(
        f"    Prediction shape: "
        f"{tuple(prediction.shape)}"
    )

    # ========================================================
    # HEATMAP
    # ========================================================

    print(
        "[10] Processing attention heatmap..."
    )

    heatmap_cpu = (
        heatmap
        .detach()
        .cpu()
    )

    heatmap_base64 = (
        heatmap_to_base64(
            heatmap
        )
    )

    scene_heatmap_base64 = (
        heatmap_to_scene_base64(
            heatmap,
            scene_width,
            scene_height
        )
    )

    print(
        f"    Heatmap shape: "
        f"{tuple(heatmap.shape)}"
    )

    print(
        f"    Min: "
        f"{float(heatmap_cpu.min()):.6f}"
    )

    print(
        f"    Max: "
        f"{float(heatmap_cpu.max()):.6f}"
    )

    print(
        f"    Mean: "
        f"{float(heatmap_cpu.mean()):.6f}"
    )

    # ========================================================
    # TEMPORAL ATTENTION
    # ========================================================

    print(
        "[11] Processing temporal attention..."
    )

    attention_cpu = (
        attention
        .detach()
        .cpu()
    )

    attention_values = (
        attention_cpu
        .flatten()
        .tolist()
    )

    print(
        f"    Attention shape: "
        f"{tuple(attention.shape)}"
    )

    print(
        f"    Attention values: "
        f"{attention_values}"
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    device = str(
        next(
            model_engine.model.parameters()
        ).device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model_engine.model.parameters()
    )

    # ========================================================
    # FACE LANDMARK INFORMATION
    #
    # IMPORTANT:
    # Do NOT access:
    #
    #     face["landmarks"]["right_eye"]
    #
    # because your current detector's landmark
    # structure is different.
    #
    # eye_features already contains the required
    # eye/nose coordinates.
    # ========================================================

    right_eye = eye_features.get(
        "right_eye",
        [0.0, 0.0]
    )

    left_eye = eye_features.get(
        "left_eye",
        [0.0, 0.0]
    )

    nose = eye_features.get(
        "nose",
        [0.0, 0.0]
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    response = {

        "success":
            True,

        # ====================================================
        # MODEL
        # ====================================================

        "model": {

            "name":
                "L-DriX-Net",

            "device":
                device,

            "parameters":
                parameter_count
        },

        # ====================================================
        # ACTUAL 24-D MODEL OUTPUT
        # ====================================================

        "prediction": {

            "values":
                [
                    safe_float(value)
                    for value
                    in prediction_values
                ],

            "shape":
                list(
                    prediction.shape
                ),

            "dimension":
                len(
                    prediction_values
                )
        },

        # ====================================================
        # ACTUAL ATTENTION HEATMAP
        # ====================================================

        "attention_heatmap": {

            "shape":
                list(
                    heatmap.shape
                ),

            "min":
                float(
                    heatmap_cpu.min()
                ),

            "max":
                float(
                    heatmap_cpu.max()
                ),

            "mean":
                float(
                    heatmap_cpu.mean()
                ),

            "image_base64":
                heatmap_base64,

            "scene_width":
                int(scene_width),

            "scene_height":
                int(scene_height),

            "scene_image_base64":
                scene_heatmap_base64
        },

        # ====================================================
        # TEMPORAL ATTENTION
        # ====================================================

        "temporal_attention": {

            "values":
                [
                    safe_float(value)
                    for value
                    in attention_values
                ],

            "shape":
                list(
                    attention.shape
                )
        },

        # ====================================================
        # FACE DETECTION
        # ====================================================

        "face_detection": {

            "bbox": {

                "x":
                    int(face["x"]),

                "y":
                    int(face["y"]),

                "w":
                    int(face["w"]),

                "h":
                    int(face["h"])
            },

            "confidence":
                safe_float(
                    face["confidence"]
                ),

            "landmarks": {

                "right_eye": [

                    safe_float(
                        right_eye[0]
                    ),

                    safe_float(
                        right_eye[1]
                    )

                ],

                "left_eye": [

                    safe_float(
                        left_eye[0]
                    ),

                    safe_float(
                        left_eye[1]
                    )

                ],

                "nose": [

                    safe_float(
                        nose[0]
                    ),

                    safe_float(
                        nose[1]
                    )

                ]

            }

        },

        # ====================================================
        # EYE FEATURES
        # ====================================================

        "eye_features":
            eye_features,

        # ====================================================
        # HEAD POSE
        # ====================================================

        "head_pose":
            head_pose,

        # ====================================================
        # GAZE
        # ====================================================

        "gaze":
            gaze,

        # ====================================================
        # GAZE VECTOR
        # ====================================================

        "gaze_vector": {

            "values": [

                safe_float(
                    value
                )

                for value
                in gaze_vector

            ],

            "dimension":
                len(gaze_vector)
        },

        # ====================================================
        # FRAME INFORMATION
        # ====================================================

        "frames": {

            "driver": {

                "width":
                    int(driver_width),

                "height":
                    int(driver_height)

            },

            "scene": {

                "width":
                    int(scene_width),

                "height":
                    int(scene_height)

            }

        },

        # ====================================================
        # IMPORTANT NOTE
        # ====================================================

        "note": (
            "The live gaze vector is an approximate "
            "geometric estimate derived from webcam "
            "facial landmarks, eye features and head "
            "pose. It is not equivalent to calibrated "
            "dataset ground-truth gaze."
        )

    }

    print()
    print("=" * 70)
    print("L-DriX-Net prediction completed successfully")
    print("=" * 70)
    print()

    return response