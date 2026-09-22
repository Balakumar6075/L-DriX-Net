import cv2
import sys
import time

from backend.detection.face_detector import FaceDetector
from backend.detection.eye_state_detector import EyeStateDetector


def select_largest_face(faces):

    if not faces:
        return None

    return max(
        faces,
        key=lambda f: f["w"] * f["h"]
    )


def main():

    print("=" * 70)
    print("L-DriX-Net - RAW EYE CNN LIVE TEST")
    print("=" * 70)

    print()
    print("Loading face detector...")

    face_detector = FaceDetector()

    print("Loading eye-state CNN...")

    eye_detector = EyeStateDetector()

    print()
    print("All modules loaded.")
    print()
    print("Controls:")
    print("  Q / ESC -> Quit")
    print("=" * 70)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open camera.")

        sys.exit(1)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        # ============================================================
        # FACE DETECTION
        # ============================================================

        faces = face_detector.detect(
            frame
        )

        face = select_largest_face(
            faces
        )

        # ============================================================
        # DISPLAY
        # ============================================================

        if face is not None:

            x = int(face["x"])
            y = int(face["y"])
            w = int(face["w"])
            h = int(face["h"])

            # --------------------------------------------------------
            # FACE BOX
            # --------------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # --------------------------------------------------------
            # RAW CNN
            # --------------------------------------------------------

            result = eye_detector.predict(
                frame,
                face
            )

            if result["valid"]:

                left = result["left_eye"]
                right = result["right_eye"]

                # ====================================================
                # LEFT EYE
                # ====================================================

                left_point = face["left_eye"]

                left_x = int(
                    left_point[0]
                )

                left_y = int(
                    left_point[1]
                )

                cv2.circle(
                    frame,
                    (left_x, left_y),
                    6,
                    (255, 255, 0),
                    -1
                )

                left_text = (
                    f"L: {left['state']} "
                    f"| OPEN {left['open_probability'] * 100:.1f}% "
                    f"| CLOSED {left['closed_probability'] * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    left_text,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                # ====================================================
                # RIGHT EYE
                # ====================================================

                right_point = face["right_eye"]

                right_x = int(
                    right_point[0]
                )

                right_y = int(
                    right_point[1]
                )

                cv2.circle(
                    frame,
                    (right_x, right_y),
                    6,
                    (255, 255, 0),
                    -1
                )

                right_text = (
                    f"R: {right['state']} "
                    f"| OPEN {right['open_probability'] * 100:.1f}% "
                    f"| CLOSED {right['closed_probability'] * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    right_text,
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                # ====================================================
                # BOTH EYES
                # ====================================================

                both_closed = (
                    left["state"] == "CLOSED"
                    and
                    right["state"] == "CLOSED"
                )

                both_text = (
                    f"BOTH EYES: "
                    f"{'CLOSED' if both_closed else 'OPEN'}"
                )

                cv2.putText(
                    frame,
                    both_text,
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0)
                    if not both_closed
                    else (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Eye prediction invalid",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ============================================================
        # CONTROLS
        # ============================================================

        cv2.putText(
            frame,
            "Q / ESC = Quit",
            (30, 700),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "L-DriX-Net - Raw Eye CNN Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:

            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()