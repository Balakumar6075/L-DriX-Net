import os
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(
    CURRENT_DIR
)

PROJECT_ROOT = os.path.dirname(
    BACKEND_DIR
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from backend.detection.eye_state_model import (
    EyeStateCNN
)


# ============================================================
# EYE STATE DETECTOR
# ============================================================

class EyeStateDetector:

    CLOSED = 0
    OPEN = 1

    def __init__(
        self,
        checkpoint_path=None
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if checkpoint_path is None:

            checkpoint_path = os.path.join(
                BACKEND_DIR,
                "checkpoints",
                "eye_state_best.pth"
            )

        self.checkpoint_path = (
            checkpoint_path
        )

        print(
            "Loading eye-state model..."
        )

        print(
            f"Device: {self.device}"
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = EyeStateCNN(
            pretrained=False
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # ----------------------------------------------------
        # MAKE SURE THIS IS THE OLD 2-CLASS MODEL
        # ----------------------------------------------------

        if checkpoint.get(
            "num_classes",
            2
        ) != 2:

            raise RuntimeError(
                "\nThe current checkpoint is NOT "
                "the original 2-class checkpoint.\n\n"
                "Expected:\n"
                "  0 -> closed\n"
                "  1 -> open\n\n"
                "The current checkpoint appears "
                "to be the new 3-class model.\n\n"
                "Restore/retrain the original "
                "2-class checkpoint first."
            )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        self.transform = transforms.Compose([

            transforms.ToPILImage(),

            transforms.Resize(
                (128, 128)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        print()
        print(
            "Class mapping:"
        )

        print(
            "  0 -> CLOSED"
        )

        print(
            "  1 -> OPEN"
        )

        print()
        print(
            "Eye-state model loaded successfully."
        )

        print(
            f"Checkpoint: "
            f"{self.checkpoint_path}"
        )

    # ========================================================
    # CROP EYE
    # ========================================================

    def crop_eye(
        self,
        frame,
        eye_point,
        face_w,
        face_h
    ):

        x = int(
            round(
                float(
                    eye_point[0]
                )
            )
        )

        y = int(
            round(
                float(
                    eye_point[1]
                )
            )
        )

        crop_w = max(
            20,
            int(
                face_w * 0.30
            )
        )

        crop_h = max(
            15,
            int(
                face_h * 0.18
            )
        )

        x1 = max(
            0,
            x - crop_w // 2
        )

        y1 = max(
            0,
            y - crop_h // 2
        )

        x2 = min(
            frame.shape[1],
            x + crop_w // 2
        )

        y2 = min(
            frame.shape[0],
            y + crop_h // 2
        )

        if x2 <= x1 or y2 <= y1:

            return None

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:

            return None

        return crop

    # ========================================================
    # SINGLE EYE
    # ========================================================

    def predict_eye(
        self,
        crop
    ):

        if crop is None:

            return {

                "valid": False,

                "state": "UNKNOWN",

                "confidence": 0.0,

                "open_probability": 0.0,

                "closed_probability": 0.0
            }

        try:

            tensor = (
                self.transform(
                    crop
                )
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():

                logits = self.model(
                    tensor
                )

                probabilities = F.softmax(
                    logits,
                    dim=1
                )[0]

            predicted_class = int(
                torch.argmax(
                    probabilities
                ).item()
            )

            confidence = float(
                probabilities[
                    predicted_class
                ].item()
            )

            closed_probability = float(
                probabilities[
                    self.CLOSED
                ].item()
            )

            open_probability = float(
                probabilities[
                    self.OPEN
                ].item()
            )

            if predicted_class == self.CLOSED:

                state = "CLOSED"

            else:

                state = "OPEN"

            return {

                "valid": True,

                "state": state,

                "confidence": round(
                    confidence,
                    4
                ),

                "open_probability": round(
                    open_probability,
                    4
                ),

                "closed_probability": round(
                    closed_probability,
                    4
                )
            }

        except Exception as e:

            print(
                f"Eye prediction error: {e}"
            )

            return {

                "valid": False,

                "state": "UNKNOWN",

                "confidence": 0.0,

                "open_probability": 0.0,

                "closed_probability": 0.0
            }

    # ========================================================
    # BOTH EYES
    # ========================================================

    def predict(
        self,
        frame,
        face
    ):

        if face is None:

            return {

                "valid": False,

                "eyes_closed": False,

                "left_eye": {
                    "valid": False,
                    "state": "UNKNOWN"
                },

                "right_eye": {
                    "valid": False,
                    "state": "UNKNOWN"
                }
            }

        face_w = float(
            face["w"]
        )

        face_h = float(
            face["h"]
        )

        # ----------------------------------------------------
        # CROPS
        # ----------------------------------------------------

        left_crop = self.crop_eye(
            frame,
            face["left_eye"],
            face_w,
            face_h
        )

        right_crop = self.crop_eye(
            frame,
            face["right_eye"],
            face_w,
            face_h
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        left_result = self.predict_eye(
            left_crop
        )

        right_result = self.predict_eye(
            right_crop
        )

        # ----------------------------------------------------
        # BOTH CLOSED
        # ----------------------------------------------------

        eyes_closed = (

            left_result["state"]
            == "CLOSED"

            and

            right_result["state"]
            == "CLOSED"
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {

            "valid": (
                left_result["valid"]
                and
                right_result["valid"]
            ),

            "eyes_closed":
                eyes_closed,

            "left_eye":
                left_result,

            "right_eye":
                right_result
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print(
        "L-DriX-Net 2-Class CNN Eye-State Detector"
    )
    print("=" * 65)

    detector = EyeStateDetector()

    print()
    print(
        "Detector initialized successfully."
    )

    print()
    print(
        "Classes:"
    )

    print(
        "  0 -> CLOSED"
    )

    print(
        "  1 -> OPEN"
    )

    print("=" * 65)