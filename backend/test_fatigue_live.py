import cv2
import time
import sys

from detection.face_detector import FaceDetector
from detection.eye_state_detector import EyeStateDetector
from detection.eye_state_temporal import TemporalEyeStateProcessor
from detection.fatigue_progression import FatigueProgressionTracker


# ============================================================
# COLORS
# ============================================================

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)


# ============================================================
# FACE SELECTION
# ============================================================

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


# ============================================================
# FATIGUE BAR
# ============================================================

def draw_fatigue_bar(frame, fatigue):

    x = frame.shape[1] - 70

    y_top = 100
    y_bottom = 550

    bar_width = 35

    bar_height = (
        y_bottom - y_top
    )

    fatigue = max(
        0.0,
        min(
            100.0,
            float(fatigue)
        )
    )

    # --------------------------------------------------------
    # BORDER
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x, y_top),
        (x + bar_width, y_bottom),
        BLACK,
        2
    )

    # --------------------------------------------------------
    # FILL
    # --------------------------------------------------------

    filled_height = int(
        bar_height *
        (fatigue / 100.0)
    )

    if filled_height > 0:

        fill_top = (
            y_bottom
            - filled_height
        )

        cv2.rectangle(
            frame,
            (
                x + 2,
                fill_top
            ),
            (
                x + bar_width - 2,
                y_bottom - 2
            ),
            RED,
            -1
        )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "0",
        (
            x - 25,
            y_bottom
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        BLACK,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "50",
        (
            x - 30,
            (y_top + y_bottom) // 2
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        BLACK,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "100",
        (
            x - 40,
            y_top + 5
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        BLACK,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "FATIGUE",
        (
            x - 25,
            y_top - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BLACK,
        1,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print(
        "L-DriX-Net LIVE FATIGUE PROGRESSION TEST"
    )
    print("=" * 65)

    print()
    print(
        "Loading Face Detector..."
    )

    face_detector = FaceDetector()

    print(
        "Loading Eye-State CNN..."
    )

    eye_detector = EyeStateDetector()

    print(
        "Loading Temporal Eye Processor..."
    )

    temporal_processor = (
        TemporalEyeStateProcessor()
    )

    print(
        "Loading Fatigue Tracker..."
    )

    fatigue_tracker = (
        FatigueProgressionTracker()
    )

    print()
    print(
        f"Eye CNN device: "
        f"{eye_detector.device}"
    )

    print()
    print(
        "All modules loaded."
    )

    print("=" * 65)

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

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
    print("Controls:")
    print("  Q / ESC -> Quit")
    print("  R       -> Reset fatigue")

    print("=" * 65)

    # ========================================================
    # FPS
    # ========================================================

    previous_time = time.monotonic()

    fps = 0.0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        now = time.monotonic()

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed_frame = (
            now
            - previous_time
        )

        previous_time = now

        if elapsed_frame > 0:

            current_fps = (
                1.0
                / elapsed_frame
            )

            if fps == 0:

                fps = current_fps

            else:

                fps = (
                    0.9 * fps
                    +
                    0.1 * current_fps
                )

        # ====================================================
        # FACE DETECTION
        # ====================================================

        faces = face_detector.detect(
            frame
        )

        face = select_largest_face(
            faces
        )

        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        eye_state = "UNKNOWN"

        blink_count = (
            temporal_processor.total_blinks
        )

        closure_duration = 0.0

        prolonged_closure = False

        # ----------------------------------------------------
        # DEFAULT HEAD POSE
        #
        # The fatigue tracker currently uses neutral values
        # in this live eye/fatigue test.
        # ----------------------------------------------------

        head_pose = {
            "yaw": 0.0,
            "pitch": 0.0
        }

        # ----------------------------------------------------
        # DEFAULT FATIGUE UPDATE
        #
        # IMPORTANT:
        # FatigueProgressionTracker does NOT accept eyes_valid.
        # ----------------------------------------------------

        fatigue_result = (
            fatigue_tracker.update(
                eyes_closed=False,
                head_pose=head_pose,
                gaze={},
                now=now
            )
        )

        # ====================================================
        # FACE FOUND
        # ====================================================

        if face is not None:

            x = int(face["x"])
            y = int(face["y"])
            w = int(face["w"])
            h = int(face["h"])

            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                GREEN,
                2
            )

            # ------------------------------------------------
            # EYE CNN
            # ------------------------------------------------

            eye_result = (
                eye_detector.predict(
                    frame,
                    face
                )
            )

            if eye_result["valid"]:

                left_result = (
                    eye_result["left_eye"]
                )

                right_result = (
                    eye_result["right_eye"]
                )

                # ------------------------------------------------
                # TEMPORAL PROCESSOR
                # ------------------------------------------------

                temporal_result = (
                    temporal_processor.update(
                        left_result,
                        right_result,
                        now
                    )
                )

                eye_state = (
                    temporal_result["state"]
                )

                blink_count = (
                    temporal_result[
                        "total_blinks"
                    ]
                )

                closure_duration = (
                    temporal_result[
                        "closure_duration"
                    ]
                )

                prolonged_closure = (
                    temporal_result[
                        "prolonged_closure"
                    ]
                )

                # ------------------------------------------------
                # FATIGUE
                # ------------------------------------------------
                #
                # IMPORTANT:
                # The restored fatigue tracker accepts:
                #
                # eyes_closed
                # head_pose
                # gaze
                # now
                #
                # It does NOT accept eyes_valid.
                # ------------------------------------------------

                fatigue_result = (
                    fatigue_tracker.update(
                        eyes_closed=(
                            temporal_result[
                                "eyes_closed"
                            ]
                        ),
                        head_pose=head_pose,
                        gaze={},
                        now=now
                    )
                )

                # =================================================
                # LEFT EYE
                # =================================================

                left_point = face[
                    "left_eye"
                ]

                left_x = int(
                    left_point[0]
                )

                left_y = int(
                    left_point[1]
                )

                cv2.circle(
                    frame,
                    (
                        left_x,
                        left_y
                    ),
                    6,
                    CYAN,
                    -1
                )

                left_label = (
                    f"L: "
                    f"{left_result['state']} "
                    f"{left_result['confidence'] * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    left_label,
                    (
                        left_x - 80,
                        left_y - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    BLACK,
                    1,
                    cv2.LINE_AA
                )

                # =================================================
                # RIGHT EYE
                # =================================================

                right_point = face[
                    "right_eye"
                ]

                right_x = int(
                    right_point[0]
                )

                right_y = int(
                    right_point[1]
                )

                cv2.circle(
                    frame,
                    (
                        right_x,
                        right_y
                    ),
                    6,
                    CYAN,
                    -1
                )

                right_label = (
                    f"R: "
                    f"{right_result['state']} "
                    f"{right_result['confidence'] * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    right_label,
                    (
                        right_x - 80,
                        right_y - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    BLACK,
                    1,
                    cv2.LINE_AA
                )

        # ====================================================
        # FATIGUE VALUES
        # ====================================================

        fatigue = (
            fatigue_result[
                "fatigue_index"
            ]
        )

        trend = (
            fatigue_result[
                "trend"
            ]
        )

        status = (
            fatigue_result[
                "status"
            ]
        )

        focused_duration = (
            fatigue_result[
                "focused_duration"
            ]
        )

        # ====================================================
        # INFORMATION
        # ====================================================

        cv2.putText(
            frame,
            f"Eye State: {eye_state}",
            (
                20,
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            BLACK,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Fatigue: {fatigue:.1f} / 100",
            (
                20,
                80
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            BLACK,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Status: {status}",
            (
                20,
                115
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            BLACK,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Trend: {trend}",
            (
                20,
                150
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BLACK,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (
                20,
                185
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BLACK,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Eye Closure: {closure_duration:.2f}s",
            (
                20,
                220
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BLACK,
            2,
            cv2.LINE_AA
        )

        # --------------------------------------------------------
        # PROLONGED CLOSURE
        # --------------------------------------------------------

        if prolonged_closure:

            cv2.putText(
                frame,
                "PROLONGED EYE CLOSURE",
                (
                    20,
                    260
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                RED,
                2,
                cv2.LINE_AA
            )

        # --------------------------------------------------------
        # FOCUSED TIME
        # --------------------------------------------------------

        focused_minutes = (
            focused_duration / 60.0
        )

        cv2.putText(
            frame,
            f"Focused: {focused_minutes:.1f} min",
            (
                20,
                295
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BLACK,
            2,
            cv2.LINE_AA
        )

        # --------------------------------------------------------
        # FPS
        # --------------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (
                20,
                330
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BLACK,
            1,
            cv2.LINE_AA
        )

        # ========================================================
        # FATIGUE BAR
        # ========================================================

        draw_fatigue_bar(
            frame,
            fatigue
        )

        # ========================================================
        # CONTROLS
        # ========================================================

        cv2.putText(
            frame,
            "R = Reset   Q = Quit",
            (
                20,
                frame.shape[0] - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BLACK,
            1,
            cv2.LINE_AA
        )

        # ========================================================
        # DISPLAY
        # ========================================================

        cv2.imshow(
            "L-DriX-Net - Live Fatigue Progression",
            frame
        )

        # ========================================================
        # KEYBOARD
        # ========================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if (
            key == ord("q")
            or key == 27
        ):

            break

        elif key == ord("r"):

            fatigue_tracker.reset()

            temporal_processor.reset()

            print(
                "Fatigue and eye-state history reset."
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 65)
    print(
        "Live fatigue test stopped."
    )
    print("=" * 65)


if __name__ == "__main__":

    main()