import os

from PIL import Image, ImageDraw


SUBJECT_PATH = "dataset/Subject01_1_data"

FACE_DIR = os.path.join(
    SUBJECT_PATH,
    "face_ims"
)

SCENE_DIR = os.path.join(
    SUBJECT_PATH,
    "scene_ims"
)

FRAME_IDS = [
    190, 193, 223, 226,
    229, 238, 241, 244,
    250, 253, 256, 286,
    304, 307, 310, 313
]


def main():

    print("=" * 60)
    print("Temporal Sequence Visualization")
    print("=" * 60)

    cell_width = 300
    cell_height = 250

    columns = 4
    rows = 4

    canvas = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height
        ),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for index, frame_id in enumerate(FRAME_IDS):

        face_path = os.path.join(
            FACE_DIR,
            f"{frame_id:08d}_face.png"
        )

        scene_path = os.path.join(
            SCENE_DIR,
            f"{frame_id:08d}_scene.png"
        )

        face = Image.open(
            face_path
        ).convert("RGB")

        scene = Image.open(
            scene_path
        ).convert("RGB")

        # Resize images for visualization

        face.thumbnail(
            (140, 180)
        )

        scene.thumbnail(
            (140, 180)
        )

        x = (
            index % columns
        ) * cell_width

        y = (
            index // columns
        ) * cell_height

        # Frame label

        draw.text(
            (x + 10, y + 5),
            f"Frame {frame_id}",
            fill="black"
        )

        # Face

        canvas.paste(
            face,
            (
                x + 5,
                y + 35
            )
        )

        # Scene

        canvas.paste(
            scene,
            (
                x + 150,
                y + 35
            )
        )

    output_path = (
        "temporal_sequence_preview.jpg"
    )

    canvas.save(
        output_path,
        quality=95
    )

    print()
    print(
        "Saved:",
        output_path
    )

    print("=" * 60)


if __name__ == "__main__":
    main()