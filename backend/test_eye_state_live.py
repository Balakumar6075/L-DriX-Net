import cv2
import sys

from detection.face_detector import FaceDetector
from detection.eye_detection import EyeDetector


def select_largest_face(faces):
    """
    YuNet returns multiple detected faces as a list.

    Select the face with the largest bounding-box area.
    This is appropriate for a driver-camera scenario where
    the driver's face should normally be the largest face.
    """

    if faces is None:
        return None

    if not isinstance(faces, list):
        return faces

    if len(faces) == 0:
        return None

    largest_face = max(
        faces,
        key=lambda face: (
            float(face["w"]) * float(face["h"])
        )
    )

    return largest_face


def main():

    print("=" * 60)
    print("L-DriX-Net Live Eye-State Test")
    print("=" * 60)

    # ---------------------------------------------------------
    # INITIALIZE DETECTORS
    # ---------------------------------------------------------

    print("Initializing face detector...")

    face_detector = FaceDetector()

    print("Initializing eye detector...")

    eye_detector = EyeDetector()

    # ---------------------------------------------------------
    # OPEN CAMERA
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print()
        print("ERROR: Could not open camera.")
        print()
        print("Possible reasons:")
        print("1. Camera is already being used by another application.")
        print("2. Camera permission is disabled.")
        print("3. Camera index is incorrect.")
        print()

        sys.exit(1)

    # Request a reasonable camera resolution.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera opened successfully.")
    print()

    print("Controls:")
    print("  Q   -> Quit")
    print("  ESC -> Quit")
    print("=" * 60)

    # ---------------------------------------------------------
    # MAIN CAMERA LOOP
    # ---------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        # -----------------------------------------------------
        # FACE DETECTION
        # -----------------------------------------------------

        faces = face_detector.detect(frame)

        # YuNet returns a list of detected faces.
        # Select the largest one.
        face = select_largest_face(faces)

        # -----------------------------------------------------
        # EYE DETECTION
        # -----------------------------------------------------

        eye_features = None

        if face is not None:

            eye_features = eye_detector.extract(
                face,
                frame
            )

        # -----------------------------------------------------
        # FACE FOUND
        # -----------------------------------------------------

        if face is not None and eye_features is not None:

            # -------------------------------------------------
            # DRAW FACE BOUNDING BOX
            # -------------------------------------------------

            x = int(face["x"])
            y = int(face["y"])
            w = int(face["w"])
            h = int(face["h"])

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # -------------------------------------------------
            # DRAW EYE FEATURES
            # -------------------------------------------------

            frame = eye_detector.draw_features(
                frame,
                eye_features
            )

            # -------------------------------------------------
            # GET EYE STATES
            # -------------------------------------------------

            right_open = eye_features.get(
                "right_eye_open",
                False
            )

            left_open = eye_features.get(
                "left_eye_open",
                False
            )

            eyes_closed = eye_features.get(
                "eyes_closed",
                False
            )

            # -------------------------------------------------
            # GET ANALYSIS VALUES
            # -------------------------------------------------

            right_analysis = eye_features.get(
                "right_eye_analysis",
                {}
            )

            left_analysis = eye_features.get(
                "left_eye_analysis",
                {}
            )

            right_dark = right_analysis.get(
                "dark_ratio",
                0.0
            )

            right_std = right_analysis.get(
                "local_std",
                0.0
            )

            right_mean = right_analysis.get(
                "mean_intensity",
                0.0
            )

            left_dark = left_analysis.get(
                "dark_ratio",
                0.0
            )

            left_std = left_analysis.get(
                "local_std",
                0.0
            )

            left_mean = left_analysis.get(
                "mean_intensity",
                0.0
            )

            # -------------------------------------------------
            # DISPLAY RIGHT EYE
            # -------------------------------------------------

            cv2.putText(
                frame,
                (
                    "Right Eye: "
                    + (
                        "OPEN"
                        if right_open
                        else "CLOSED"
                    )
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # DISPLAY LEFT EYE
            # -------------------------------------------------

            cv2.putText(
                frame,
                (
                    "Left Eye: "
                    + (
                        "OPEN"
                        if left_open
                        else "CLOSED"
                    )
                ),
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # DISPLAY OVERALL EYE STATE
            # -------------------------------------------------

            cv2.putText(
                frame,
                (
                    "Overall: "
                    + (
                        "EYES CLOSED"
                        if eyes_closed
                        else "EYES OPEN"
                    )
                ),
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # RIGHT EYE ANALYSIS
            # -------------------------------------------------

            cv2.putText(
                frame,
                (
                    f"R Dark: {right_dark:.2f}  "
                    f"Std: {right_std:.1f}  "
                    f"Mean: {right_mean:.1f}"
                ),
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # LEFT EYE ANALYSIS
            # -------------------------------------------------

            cv2.putText(
                frame,
                (
                    f"L Dark: {left_dark:.2f}  "
                    f"Std: {left_std:.1f}  "
                    f"Mean: {left_mean:.1f}"
                ),
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # FACE CONFIDENCE
            # -------------------------------------------------

            confidence = face.get(
                "confidence",
                None
            )

            if confidence is not None:

                cv2.putText(
                    frame,
                    f"Face Confidence: {float(confidence):.2f}",
                    (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # -----------------------------------------------------
        # NO FACE FOUND
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
        # CAMERA WINDOW
        # -----------------------------------------------------

        cv2.imshow(
            "L-DriX-Net - Eye State Test",
            frame
        )

        # -----------------------------------------------------
        # KEYBOARD INPUT
        # -----------------------------------------------------

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
    print("Eye-state test stopped.")
    print("=" * 60)


if __name__ == "__main__":
    main()