import cv2
import sys

from detection.face_detector import FaceDetector
from detection.eye_state_detector import EyeStateDetector


def select_largest_face(faces):
    """
    Select the largest detected face.
    """

    if faces is None:
        return None

    if not isinstance(faces, list):
        return faces

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda face:
        float(face["w"]) * float(face["h"])
    )


def draw_eye_box(
    frame,
    eye_point,
    face_w,
    face_h,
    state,
    confidence
):
    """
    Draw the eye crop region and prediction.
    """

    x = int(
        round(
            float(eye_point[0])
        )
    )

    y = int(
        round(
            float(eye_point[1])
        )
    )

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

    # Draw crop rectangle
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2
    )

    # Draw prediction above eye
    text = (
        f"{state} "
        f"{confidence * 100:.1f}%"
    )

    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )


def main():

    print("=" * 60)
    print("L-DriX-Net LIVE CNN Eye-State Test")
    print("=" * 60)

    # ---------------------------------------------------------
    # INITIALIZE
    # ---------------------------------------------------------

    print()
    print("Loading face detector...")

    face_detector = FaceDetector()

    print(
        "Loading trained eye-state CNN..."
    )

    eye_detector = EyeStateDetector()

    print()
    print(
        f"Eye CNN device: "
        f"{eye_detector.device}"
    )

    print()
    print(
        "Models loaded successfully."
    )

    # ---------------------------------------------------------
    # CAMERA
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print()
        print(
            "ERROR: Could not open camera."
        )

        sys.exit(1)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print()
    print(
        "Camera opened successfully."
    )

    print()
    print(
        "Controls:"
    )

    print(
        "  Q   -> Quit"
    )

    print(
        "  ESC -> Quit"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # CAMERA LOOP
    # ---------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Failed to read camera frame."
            )

            break

        # -----------------------------------------------------
        # FACE DETECTION
        # -----------------------------------------------------

        faces = face_detector.detect(
            frame
        )

        face = select_largest_face(
            faces
        )

        # -----------------------------------------------------
        # FACE FOUND
        # -----------------------------------------------------

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

            # Face bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # -------------------------------------------------
            # CNN EYE PREDICTION
            # -------------------------------------------------

            result = eye_detector.predict(
                frame,
                face
            )

            if result["valid"]:

                left_eye = result[
                    "left_eye"
                ]

                right_eye = result[
                    "right_eye"
                ]

                eyes_closed = result[
                    "eyes_closed"
                ]

                # -------------------------------------------------
                # LEFT EYE
                # -------------------------------------------------

                draw_eye_box(
                    frame,
                    face["left_eye"],
                    w,
                    h,
                    left_eye["state"],
                    left_eye["confidence"]
                )

                # -------------------------------------------------
                # RIGHT EYE
                # -------------------------------------------------

                draw_eye_box(
                    frame,
                    face["right_eye"],
                    w,
                    h,
                    right_eye["state"],
                    right_eye["confidence"]
                )

                # -------------------------------------------------
                # OVERALL STATE
                # -------------------------------------------------

                if eyes_closed:

                    overall_text = (
                        "EYES CLOSED"
                    )

                else:

                    overall_text = (
                        "EYES OPEN"
                    )

                cv2.putText(
                    frame,
                    overall_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # -------------------------------------------------
                # LEFT PROBABILITIES
                # -------------------------------------------------

                left_open = (
                    left_eye[
                        "open_probability"
                    ]
                    * 100
                )

                left_closed = (
                    left_eye[
                        "closed_probability"
                    ]
                    * 100
                )

                cv2.putText(
                    frame,
                    (
                        f"Left  "
                        f"O:{left_open:.1f}% "
                        f"C:{left_closed:.1f}%"
                    ),
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # -------------------------------------------------
                # RIGHT PROBABILITIES
                # -------------------------------------------------

                right_open = (
                    right_eye[
                        "open_probability"
                    ]
                    * 100
                )

                right_closed = (
                    right_eye[
                        "closed_probability"
                    ]
                    * 100
                )

                cv2.putText(
                    frame,
                    (
                        f"Right "
                        f"O:{right_open:.1f}% "
                        f"C:{right_closed:.1f}%"
                    ),
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # -------------------------------------------------
                # MODEL
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    "Eye CNN: MobileNetV3-Small",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            else:

                cv2.putText(
                    frame,
                    "Eye prediction unavailable",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

        # -----------------------------------------------------
        # NO FACE
        # -----------------------------------------------------

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        # -----------------------------------------------------
        # DISPLAY
        # -----------------------------------------------------

        cv2.imshow(
            "L-DriX-Net - CNN Eye State",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:

            break

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("Live CNN eye-state test stopped.")
    print("=" * 60)


if __name__ == "__main__":

    main()