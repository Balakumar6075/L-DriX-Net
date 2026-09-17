# ============================================================
# L-DriX-Net
# Real Inference Pipeline Test
# ============================================================

from pathlib import Path
import sys

import torch
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

sys.path.insert(
    0,
    str(PROJECT_DIR)
)


# ============================================================
# IMPORT INFERENCE
# ============================================================

from inference import (
    LDriXNetInference
)


# ============================================================
# TEST IMAGE SEARCH
# ============================================================

def find_test_image():

    # Try to locate an image from common
    # project directories.

    possible_directories = [
        PROJECT_DIR / "face_ims",
        PROJECT_DIR / "scene_ims",
        PROJECT_DIR / "outputs",
        PROJECT_DIR / "data",
    ]

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ]

    for directory in possible_directories:

        if not directory.exists():
            continue

        for extension in extensions:

            images = list(
                directory.glob(
                    f"*{extension}"
                )
            )

            if images:
                return images[0]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("L-DriX-Net REAL INFERENCE TEST")
    print("=" * 60)


    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    inference = (
        LDriXNetInference()
    )


    # --------------------------------------------------------
    # Find an image
    # --------------------------------------------------------

    test_image_path = (
        find_test_image()
    )

    if test_image_path is None:

        print()
        print(
            "No test image was found."
        )

        print()
        print(
            "Please place a JPG/PNG image in:"
        )

        print(
            "face_ims/"
        )

        print(
            "or"
        )

        print(
            "scene_ims/"
        )

        print()

        return


    print()
    print(
        f"Test image: {test_image_path}"
    )


    # --------------------------------------------------------
    # Use the same image temporarily for both
    # face and scene input.
    #
    # This is ONLY to verify that the neural
    # network forward pass works.
    # --------------------------------------------------------

    face_image = Image.open(
        test_image_path
    ).convert("RGB")

    scene_image = Image.open(
        test_image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Temporary 24-dimensional gaze vector
    #
    # IMPORTANT:
    # This is NOT the final live-camera gaze input.
    #
    # It only verifies that the trained model
    # accepts the expected [1, 24] tensor.
    # --------------------------------------------------------

    gaze = torch.zeros(
        24,
        dtype=torch.float32
    )


    print()
    print(
        "Running L-DriX-Net forward pass..."
    )


    # --------------------------------------------------------
    # Run actual model
    # --------------------------------------------------------

    prediction, heatmap, attention = (
        inference.predict(
            face_image,
            scene_image,
            gaze
        )
    )


    # --------------------------------------------------------
    # Print outputs
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("INFERENCE SUCCESSFUL")
    print("=" * 60)


    print()
    print(
        "Prediction:"
    )

    print(
        f"Shape: {tuple(prediction.shape)}"
    )


    print()
    print(
        "Heatmap:"
    )

    print(
        f"Shape: {tuple(heatmap.shape)}"
    )


    print()
    print(
        "Temporal Attention:"
    )

    print(
        f"Shape: {tuple(attention.shape)}"
    )


    print()
    print(
        "Prediction values:"
    )

    print(
        prediction
    )


    print()
    print(
        "Heatmap range:"
    )

    print(
        f"Min: {heatmap.min().item():.6f}"
    )

    print(
        f"Max: {heatmap.max().item():.6f}"
    )


    print()
    print("=" * 60)
    print(
        "REAL L-DriX-Net INFERENCE TEST PASSED"
    )
    print("=" * 60)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()