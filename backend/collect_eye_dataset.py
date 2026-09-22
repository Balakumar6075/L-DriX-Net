import cv2
import os
import time

from detection.face_detector import FaceDetector


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__),
    "eye_state_dataset"
)

OPEN_DIR = os.path.join(
    DATASET_ROOT,
    "open"
)

CLOSED_DIR = os.path.join(
    DATASET_ROOT,
    "closed"
)

OCCLUDED_DIR = os.path.join(
    DATASET_ROOT,
    "occluded"
)

os.makedirs(
    OPEN_DIR,
    exist_ok=True
)

os.makedirs(
    CLOSED_DIR,
    exist_ok=True
)

os.makedirs(
    OCCLUDED_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# EYE CROP FUNCTION
# ---------------------------------------------------------

def crop_eye(
    frame,
    eye_point,
    face_w,
    face_h
):

    x = int(round(float(eye_point[0])))
    y = int(round(float(eye_point[1])))

    crop_w = max(
        20,
        int(face_w * 0.30)
    )

    crop_h = max(
        15,
        int(face_h * 0.18)
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


# ---------------------------------------------------------
# SELECT LARGEST FACE
# ---------------------------------------------------------

def select_largest_face(faces):

    if faces is None:
        return None

    if not isinstance(faces, list):
        return faces

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda face:
        float(face["w"]) *
        float(face["h"])
    )


# ---------------------------------------------------------
# SAVE EYE IMAGE
# ---------------------------------------------------------

def save_eye(
    crop,
    directory,
    prefix
):

    if crop is None:
        return None

    existing = [
        f
        for f in os.listdir(directory)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    index = len(existing)

    filename = (
        f"{prefix}_{index:06d}.jpg"
    )

    path = os.path.join(
        directory,
        filename
    )

    crop = cv2.resize(
        crop,
        (128, 64)
    )

    cv2.imwrite(
        path,
        crop
    )

    return path


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print(
        "L-DriX-Net Eye-State Dataset Collector"
    )
    print("=" * 65)

    print()
    print("Dataset:")
    print(DATASET_ROOT)

    print()
    print("Classes:")
    print("  OPEN     -> normal visible eye")
    print("  CLOSED   -> eye genuinely closed")
    print("  OCCLUDED -> eye blocked by hand/object")
    print()

    print("Controls:")
    print("  O -> collect OPEN samples")
    print("  C -> collect CLOSED samples")
    print("  X -> collect OCCLUDED samples")
    print("  Q -> quit")

    print()
    print("Collection examples:")
    print()
    print("OPEN:")
    print("  - Look normally at the camera")
    print("  - Natural head movement")
    print()
    print("CLOSED:")
    print("  - Close both eyes naturally")
    print("  - Keep eyes closed")
    print()
    print("OCCLUDED:")
    print("  - Cover one eye with your hand")
    print("  - Cover both eyes")
    print("  - Put fingers partially over eyes")
    print("  - Move hand across the eye")
    print("  - Pretend to clean dust from the eye")
    print()
    print("=" * 65)

    # -----------------------------------------------------
    # FACE DETECTOR
    # -----------------------------------------------------

    face_detector = FaceDetector()

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    # -----------------------------------------------------
    # COLLECTION STATE
    # -----------------------------------------------------

    collecting = None

    last_capture = 0.0

    capture_interval = 0.08

    open_count = 0

    closed_count = 0

    occluded_count = 0

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        # -------------------------------------------------
        # FACE DETECTION
        # -------------------------------------------------

        faces = face_detector.detect(
            frame
        )

        face = select_largest_face(
            faces
        )

        status = "IDLE"

        # -------------------------------------------------
        # FACE FOUND
        # -------------------------------------------------

        if face is not None:

            x = int(
                face["x"]
            )

            y = int(
                face["y"]
            )

            w = int(
                face["w"]
            )

            h = int(
                face["h"]
            )

            cv2.rectangle(
                frame,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                (0, 255, 0),
                2
            )

            # -------------------------------------------------
            # EYE POINTS
            # -------------------------------------------------

            right_eye = face[
                "right_eye"
            ]

            left_eye = face[
                "left_eye"
            ]

            # -------------------------------------------------
            # EYE CROPS
            # -------------------------------------------------

            right_crop = crop_eye(
                frame,
                right_eye,
                w,
                h
            )

            left_crop = crop_eye(
                frame,
                left_eye,
                w,
                h
            )

            # -------------------------------------------------
            # DRAW EYE CENTERS
            # -------------------------------------------------

            cv2.circle(
                frame,
                (
                    int(right_eye[0]),
                    int(right_eye[1])
                ),
                5,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                (
                    int(left_eye[0]),
                    int(left_eye[1])
                ),
                5,
                (0, 0, 255),
                -1
            )

            # -------------------------------------------------
            # COLLECTION
            # -------------------------------------------------

            now = time.monotonic()

            if collecting is not None:

                if (
                    now - last_capture
                    >= capture_interval
                ):

                    # =========================================
                    # OPEN
                    # =========================================

                    if collecting == "open":

                        save_eye(
                            right_crop,
                            OPEN_DIR,
                            "right"
                        )

                        save_eye(
                            left_crop,
                            OPEN_DIR,
                            "left"
                        )

                        open_count += 2

                    # =========================================
                    # CLOSED
                    # =========================================

                    elif collecting == "closed":

                        save_eye(
                            right_crop,
                            CLOSED_DIR,
                            "right"
                        )

                        save_eye(
                            left_crop,
                            CLOSED_DIR,
                            "left"
                        )

                        closed_count += 2

                    # =========================================
                    # OCCLUDED
                    # =========================================

                    elif collecting == "occluded":

                        save_eye(
                            right_crop,
                            OCCLUDED_DIR,
                            "right"
                        )

                        save_eye(
                            left_crop,
                            OCCLUDED_DIR,
                            "left"
                        )

                        occluded_count += 2

                    last_capture = now

                status = (
                    "COLLECTING "
                    + collecting.upper()
                )

        # -----------------------------------------------------
        # DISPLAY
        # -----------------------------------------------------

        cv2.putText(
            frame,
            f"Mode: {status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Open samples: {open_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Closed samples: {closed_count}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Occluded samples: {occluded_count}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "O=OPEN  C=CLOSED  X=OCCLUDED  Q=QUIT",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "L-DriX-Net Eye Dataset Collector",
            frame
        )

        # -----------------------------------------------------
        # KEYBOARD
        # -----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("o"):

            collecting = "open"

            print(
                "Collecting OPEN samples..."
            )

        elif key == ord("c"):

            collecting = "closed"

            print(
                "Collecting CLOSED samples..."
            )

        elif key == ord("x"):

            collecting = "occluded"

            print(
                "Collecting OCCLUDED samples..."
            )

        elif key == ord("q") or key == 27:

            break

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 65)
    print("Dataset collection completed.")
    print("=" * 65)

    print(
        f"Open samples: {open_count}"
    )

    print(
        f"Closed samples: {closed_count}"
    )

    print(
        f"Occluded samples: {occluded_count}"
    )

    print()
    print(
        f"Open directory: {OPEN_DIR}"
    )

    print(
        f"Closed directory: {CLOSED_DIR}"
    )

    print(
        f"Occluded directory: {OCCLUDED_DIR}"
    )

    print("=" * 65)


if __name__ == "__main__":

    main()