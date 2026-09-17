# ============================================================
# L-DriX-Net
# Backend Model Inference
# ============================================================

from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

# Current file:
#
# L-DriX-Net/
# └── backend/
#     └── inference.py
#
# Therefore the project root is one level above backend.

BACKEND_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BACKEND_DIR.parent


# IMPORTANT:
# Add the project root BEFORE importing anything
# from the project's "models" package.

sys.path.insert(
    0,
    str(PROJECT_DIR)
)


# ============================================================
# NOW IMPORT PROJECT/EXTERNAL MODULES
# ============================================================

import torch

from PIL import Image

from torchvision import transforms


# ============================================================
# CHECKPOINT PATH
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "best_model.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
    ]
)


# ============================================================
# L-DriX-Net INFERENCE CLASS
# ============================================================

class LDriXNetInference:

    def __init__(self):

        print()
        print("=" * 60)
        print("L-DriX-Net Backend")
        print("=" * 60)

        print(
            f"Project: {PROJECT_DIR}"
        )

        print(
            f"Device: {DEVICE}"
        )

        print(
            f"Checkpoint: {CHECKPOINT_PATH}"
        )

        self.model = self._load_model()

        print()
        print(
            "L-DriX-Net model ready."
        )

        print("=" * 60)


    # ========================================================
    # CREATE MODEL
    # ========================================================

    def _create_model(self):

        print(
            "Importing LDriXNet..."
        )

        # The project root has already been added
        # to sys.path above.

        from models.ldrixnet import (
            LDriXNet
        )

        print(
            "LDriXNet imported successfully."
        )

        model = LDriXNet()

        return model


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    def _load_model(self):

        # ----------------------------------------------------
        # Check checkpoint
        # ----------------------------------------------------

        if not CHECKPOINT_PATH.exists():

            raise FileNotFoundError(
                "\nL-DriX-Net checkpoint not found:\n"
                f"{CHECKPOINT_PATH}"
            )


        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        print()
        print(
            "Creating model..."
        )

        model = self._create_model()


        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        print(
            "Loading trained checkpoint..."
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )


        # ----------------------------------------------------
        # Extract model state dictionary
        # ----------------------------------------------------

        if (
            isinstance(
                checkpoint,
                dict
            )
            and
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

            print(
                "Found model_state_dict "
                "inside checkpoint."
            )

        else:

            state_dict = checkpoint

            print(
                "Checkpoint is a raw "
                "state dictionary."
            )


        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        print(
            "Loading model weights..."
        )

        model.load_state_dict(
            state_dict,
            strict=True
        )


        # ----------------------------------------------------
        # Move to CUDA/CPU
        # ----------------------------------------------------

        model = model.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Evaluation mode
        # ----------------------------------------------------

        model.eval()


        # ----------------------------------------------------
        # Count parameters
        # ----------------------------------------------------

        parameter_count = sum(
            parameter.numel()
            for parameter
            in model.parameters()
        )


        print(
            f"Parameters: "
            f"{parameter_count:,}"
        )


        return model


    # ========================================================
    # IMAGE → TENSOR
    # ========================================================

    def image_to_tensor(
        self,
        image
    ):

        # ----------------------------------------------------
        # Convert to PIL
        # ----------------------------------------------------

        if isinstance(
            image,
            Image.Image
        ):

            pil_image = image.convert(
                "RGB"
            )

        else:

            pil_image = Image.open(
                image
            ).convert(
                "RGB"
            )


        # ----------------------------------------------------
        # Resize + tensor
        # ----------------------------------------------------

        tensor = IMAGE_TRANSFORM(
            pil_image
        )


        # ----------------------------------------------------
        # Add batch dimension
        #
        # [3, 224, 224]
        #
        # becomes
        #
        # [1, 3, 224, 224]
        # ----------------------------------------------------

        tensor = tensor.unsqueeze(
            0
        )


        # ----------------------------------------------------
        # Move to CUDA/CPU
        # ----------------------------------------------------

        return tensor.to(
            DEVICE
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    @torch.no_grad()
    def predict(
        self,
        face_image,
        scene_image,
        gaze
    ):

        # ----------------------------------------------------
        # Convert images
        # ----------------------------------------------------

        face = self.image_to_tensor(
            face_image
        )

        scene = self.image_to_tensor(
            scene_image
        )


        # ----------------------------------------------------
        # Convert gaze to tensor
        # ----------------------------------------------------

        if not torch.is_tensor(
            gaze
        ):

            gaze = torch.tensor(
                gaze,
                dtype=torch.float32
            )


        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        if gaze.dim() == 1:

            gaze = gaze.unsqueeze(
                0
            )


        # ----------------------------------------------------
        # Move gaze to same device
        # ----------------------------------------------------

        gaze = gaze.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Validate shape
        # ----------------------------------------------------

        if tuple(gaze.shape) != (
            1,
            24
        ):

            raise ValueError(
                "Invalid gaze input shape. "
                "Expected [1, 24], "
                f"received {tuple(gaze.shape)}"
            )


        # ----------------------------------------------------
        # ACTUAL L-DriX-Net FORWARD PASS
        # ----------------------------------------------------

        (
            prediction,
            heatmap,
            attention
        ) = self.model(
            face,
            scene,
            gaze
        )


        # ----------------------------------------------------
        # Move outputs to CPU
        # ----------------------------------------------------

        prediction = (
            prediction
            .detach()
            .cpu()
        )

        heatmap = (
            heatmap
            .detach()
            .cpu()
        )

        attention = (
            attention
            .detach()
            .cpu()
        )


        return (
            prediction,
            heatmap,
            attention
        )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    try:

        inference = (
            LDriXNetInference()
        )

        print()
        print(
            "MODEL LOADING TEST PASSED"
        )

        print(
            "The trained L-DriX-Net "
            "checkpoint was loaded successfully."
        )

        print()

    except Exception as error:

        print()
        print("=" * 60)
        print("MODEL LOADING FAILED")
        print("=" * 60)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print()

        raise