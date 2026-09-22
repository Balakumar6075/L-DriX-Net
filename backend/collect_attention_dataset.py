import cv2
import csv
import os
import time


# ============================================================
# L-DriX-Net
# MULTIMODAL DATASET COLLECTOR
# ============================================================
#
# This collector is for:
#
#   1. DISTRACTED
#   2. CONTEXTUAL_ATTENTION
#   3. DROWSY
#
# Two cameras are used:
#
#   Camera 0 -> Driver camera
#   Camera 1 -> Right-side scene camera
#
# The two cameras are synchronized by frame ID.
#
# Example:
#
#   driver/000123.jpg
#   scene/000123.jpg
#
# correspond to the same capture moment.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "custom_dataset"
)


# ------------------------------------------------------------
# CAMERA INDICES
# ------------------------------------------------------------

CAMERA_DRIVER = 0

CAMERA_SCENE = 1


# ------------------------------------------------------------
# CAMERA SETTINGS
# ------------------------------------------------------------

FRAME_WIDTH = 1280

FRAME_HEIGHT = 720

CAMERA_FPS = 30


# ------------------------------------------------------------
# IMAGE QUALITY
# ------------------------------------------------------------

JPEG_QUALITY = 95


# ------------------------------------------------------------
# SESSION PREFIX
# ------------------------------------------------------------

SESSION_PREFIX = "multimodal_session_"


# ============================================================
# AVAILABLE DRIVER STATES
# ============================================================

STATE_DISTRACTED = "DISTRACTED"

STATE_CONTEXTUAL = "CONTEXTUAL_ATTENTION"

STATE_DROWSY = "DROWSY"


# ============================================================
# CREATE SESSION
# ============================================================

def create_session():

    os.makedirs(
        DATASET_ROOT,
        exist_ok=True
    )

    existing_sessions = []

    for name in os.listdir(
        DATASET_ROOT
    ):

        path = os.path.join(
            DATASET_ROOT,
            name
        )

        if (
            os.path.isdir(path)
            and name.startswith(
                SESSION_PREFIX
            )
        ):

            existing_sessions.append(
                name
            )

    session_numbers = []

    for name in existing_sessions:

        try:

            number = int(
                name.replace(
                    SESSION_PREFIX,
                    ""
                )
            )

            session_numbers.append(
                number
            )

        except ValueError:

            pass

    if session_numbers:

        next_number = (
            max(session_numbers) + 1
        )

    else:

        next_number = 1

    session_name = (
        f"{SESSION_PREFIX}"
        f"{next_number:03d}"
    )

    session_path = os.path.join(
        DATASET_ROOT,
        session_name
    )

    driver_path = os.path.join(
        session_path,
        "driver"
    )

    scene_path = os.path.join(
        session_path,
        "scene"
    )

    os.makedirs(
        driver_path,
        exist_ok=True
    )

    os.makedirs(
        scene_path,
        exist_ok=True
    )

    return (
        session_name,
        session_path,
        driver_path,
        scene_path
    )


# ============================================================
# OPEN CAMERA
# ============================================================

def open_camera(index):

    camera = cv2.VideoCapture(
        index,
        cv2.CAP_DSHOW
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    return camera


# ============================================================
# CREATE CSV
# ============================================================

def create_csv(
    csv_path
):

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow([
        "frame_id",
        "timestamp",
        "state",
        "eye_state",
        "gaze_context",
        "scene_context"
    ])

    csv_file.flush()

    return (
        csv_file,
        csv_writer
    )


# ============================================================
# FORMAT TIME
# ============================================================

def format_duration(
    seconds
):

    seconds = int(
        max(
            0,
            seconds
        )
    )

    minutes = (
        seconds // 60
    )

    remaining_seconds = (
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "L-DriX-Net MULTIMODAL DATASET COLLECTOR"
    )
    print("=" * 70)
    print()

    print(
        "This collector is for:"
    )

    print(
        "  DISTRACTED"
    )

    print(
        "  CONTEXTUAL_ATTENTION"
    )

    print(
        "  DROWSY"
    )

    print()

    print(
        "Camera configuration:"
    )

    print(
        "  Camera 0 -> DRIVER"
    )

    print(
        "  Camera 1 -> RIGHT-SIDE SCENE"
    )

    print()

    print(
        "Controls:"
    )

    print()
    print(
        "1 = DISTRACTED"
    )

    print(
        "2 = CONTEXTUAL ATTENTION"
    )

    print(
        "3 = DROWSY"
    )

    print()

    print(
        "F = Gaze FORWARD"
    )

    print(
        "L = Gaze LEFT"
    )

    print(
        "R = Gaze RIGHT"
    )

    print(
        "D = Gaze DOWN"
    )

    print()

    print(
        "W = Scene ROAD"
    )

    print(
        "V = Scene VEHICLE"
    )

    print(
        "P = Scene PEDESTRIAN"
    )

    print(
        "O = Scene OBSTACLE"
    )

    print(
        "N = Scene NONE"
    )

    print()

    print(
        "E = Toggle eye state OPEN/CLOSED"
    )

    print(
        "SPACE = Start / Stop recording"
    )

    print(
        "X = Reset labels"
    )

    print(
        "Q = Quit"
    )

    print()

    # ========================================================
    # CREATE SESSION
    # ========================================================

    (
        session_name,
        session_path,
        driver_path,
        scene_path
    ) = create_session()

    print(
        f"Session: {session_name}"
    )

    print(
        f"Location: {session_path}"
    )

    print()

    # ========================================================
    # CREATE CSV
    # ========================================================

    csv_path = os.path.join(
        session_path,
        "annotations.csv"
    )

    (
        csv_file,
        csv_writer
    ) = create_csv(
        csv_path
    )

    # ========================================================
    # OPEN DRIVER CAMERA
    # ========================================================

    driver_camera = open_camera(
        CAMERA_DRIVER
    )

    if not driver_camera.isOpened():

        print(
            "ERROR: Could not open "
            "driver camera."
        )

        csv_file.close()

        return

    # ========================================================
    # OPEN SCENE CAMERA
    # ========================================================

    scene_camera = open_camera(
        CAMERA_SCENE
    )

    if not scene_camera.isOpened():

        print(
            "ERROR: Could not open "
            "scene camera."
        )

        driver_camera.release()

        csv_file.close()

        return

    print(
        "Both cameras opened successfully."
    )

    print()

    # ========================================================
    # STATE
    # ========================================================

    recording = False

    current_state = (
        STATE_DISTRACTED
    )

    current_eye_state = (
        "UNKNOWN"
    )

    current_gaze_context = (
        "UNKNOWN"
    )

    current_scene_context = (
        "UNKNOWN"
    )

    # ========================================================
    # FRAME COUNTER
    # ========================================================

    frame_id = 0

    total_frames = 0

    # ========================================================
    # RECORDING TIME
    # ========================================================

    recording_start_time = None

    total_recording_time = 0.0

    # ========================================================
    # FPS
    # ========================================================

    fps_counter = 0

    fps_start = time.time()

    display_fps = 0.0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # READ DRIVER CAMERA
        # ----------------------------------------------------

        driver_ok, driver_frame = (
            driver_camera.read()
        )

        # ----------------------------------------------------
        # READ SCENE CAMERA
        # ----------------------------------------------------

        scene_ok, scene_frame = (
            scene_camera.read()
        )

        if not driver_ok:

            print(
                "ERROR: Driver camera "
                "frame failed."
            )

            break

        if not scene_ok:

            print(
                "ERROR: Scene camera "
                "frame failed."
            )

            break

        # ====================================================
        # FPS
        # ====================================================

        fps_counter += 1

        fps_elapsed = (
            time.time()
            -
            fps_start
        )

        if fps_elapsed >= 1.0:

            display_fps = (
                fps_counter /
                fps_elapsed
            )

            fps_counter = 0

            fps_start = time.time()

        # ====================================================
        # DISPLAY COPIES
        # ====================================================

        driver_display = (
            driver_frame.copy()
        )

        scene_display = (
            scene_frame.copy()
        )

        # ====================================================
        # RECORDING STATUS
        # ====================================================

        if recording:

            status_text = (
                "RECORDING"
            )

            status_color = (
                0,
                0,
                255
            )

        else:

            status_text = (
                "PAUSED"
            )

            status_color = (
                0,
                255,
                255
            )

        # ====================================================
        # CURRENT RECORDING TIME
        # ====================================================

        current_recording_time = (
            total_recording_time
        )

        if (
            recording
            and
            recording_start_time
            is not None
        ):

            current_recording_time += (
                time.time()
                -
                recording_start_time
            )

        # ====================================================
        # DRIVER DISPLAY
        # ====================================================

        cv2.putText(
            driver_display,
            "L-DriX-Net DRIVER",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            driver_display,
            f"STATE: {current_state}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            driver_display,
            f"EYES: {current_eye_state}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            driver_display,
            f"GAZE: {current_gaze_context}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            driver_display,
            f"STATUS: {status_text}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        cv2.putText(
            driver_display,
            f"FPS: {display_fps:.1f}",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            driver_display,
            (
                "TIME: "
                +
                format_duration(
                    current_recording_time
                )
            ),
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SCENE DISPLAY
        # ====================================================

        cv2.putText(
            scene_display,
            "L-DriX-Net RIGHT-SIDE SCENE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            scene_display,
            f"SCENE: {current_scene_context}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            scene_display,
            f"STATE: {current_state}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            scene_display,
            f"STATUS: {status_text}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        cv2.putText(
            scene_display,
            f"FRAMES: {total_frames}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ====================================================
        # RECORD FRAMES
        # ====================================================

        if recording:

            filename = (
                f"{frame_id:06d}.jpg"
            )

            # ------------------------------------------------
            # Driver image
            # ------------------------------------------------

            driver_file = os.path.join(
                driver_path,
                filename
            )

            # ------------------------------------------------
            # Scene image
            # ------------------------------------------------

            scene_file = os.path.join(
                scene_path,
                filename
            )

            # ------------------------------------------------
            # Save driver image
            # ------------------------------------------------

            cv2.imwrite(
                driver_file,
                driver_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )

            # ------------------------------------------------
            # Save scene image
            # ------------------------------------------------

            cv2.imwrite(
                scene_file,
                scene_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )

            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            timestamp = time.time()

            # ------------------------------------------------
            # Save annotation
            # ------------------------------------------------

            csv_writer.writerow([
                frame_id,
                timestamp,
                current_state,
                current_eye_state,
                current_gaze_context,
                current_scene_context
            ])

            csv_file.flush()

            # ------------------------------------------------
            # Update counters
            # ------------------------------------------------

            frame_id += 1

            total_frames += 1

        # ====================================================
        # DISPLAY WINDOWS
        # ====================================================

        cv2.imshow(
            "L-DriX-Net Driver Camera",
            driver_display
        )

        cv2.imshow(
            "L-DriX-Net Right-Side Scene",
            scene_display
        )

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        # ====================================================
        # QUIT
        # ====================================================

        if key == ord("q"):

            break

        # ====================================================
        # STATE
        # ====================================================

        elif key == ord("1"):

            current_state = (
                STATE_DISTRACTED
            )

            print(
                "State -> DISTRACTED"
            )

        elif key == ord("2"):

            current_state = (
                STATE_CONTEXTUAL
            )

            print(
                "State -> "
                "CONTEXTUAL_ATTENTION"
            )

        elif key == ord("3"):

            current_state = (
                STATE_DROWSY
            )

            print(
                "State -> DROWSY"
            )

        # ====================================================
        # EYE STATE
        # ====================================================

        elif key == ord("e"):

            if current_eye_state == "OPEN":

                current_eye_state = (
                    "CLOSED"
                )

            else:

                current_eye_state = (
                    "OPEN"
                )

            print(
                f"Eye state -> "
                f"{current_eye_state}"
            )

        # ====================================================
        # GAZE
        # ====================================================

        elif key == ord("f"):

            current_gaze_context = (
                "FORWARD"
            )

            print(
                "Gaze -> FORWARD"
            )

        elif key == ord("l"):

            current_gaze_context = (
                "LEFT"
            )

            print(
                "Gaze -> LEFT"
            )

        elif key == ord("r"):

            current_gaze_context = (
                "RIGHT"
            )

            print(
                "Gaze -> RIGHT"
            )

        elif key == ord("d"):

            current_gaze_context = (
                "DOWN"
            )

            print(
                "Gaze -> DOWN"
            )

        # ====================================================
        # SCENE
        # ====================================================

        elif key == ord("w"):

            current_scene_context = (
                "ROAD"
            )

            print(
                "Scene -> ROAD"
            )

        elif key == ord("v"):

            current_scene_context = (
                "VEHICLE"
            )

            print(
                "Scene -> VEHICLE"
            )

        elif key == ord("p"):

            current_scene_context = (
                "PEDESTRIAN"
            )

            print(
                "Scene -> PEDESTRIAN"
            )

        elif key == ord("o"):

            current_scene_context = (
                "OBSTACLE"
            )

            print(
                "Scene -> OBSTACLE"
            )

        elif key == ord("n"):

            current_scene_context = (
                "NONE"
            )

            print(
                "Scene -> NONE"
            )

        # ====================================================
        # RESET LABELS
        # ====================================================

        elif key == ord("x"):

            current_state = (
                STATE_DISTRACTED
            )

            current_eye_state = (
                "UNKNOWN"
            )

            current_gaze_context = (
                "UNKNOWN"
            )

            current_scene_context = (
                "UNKNOWN"
            )

            print(
                "All labels reset."
            )

        # ====================================================
        # START / STOP RECORDING
        # ====================================================

        elif key == 32:

            recording = not recording

            if recording:

                recording_start_time = (
                    time.time()
                )

                print()
                print(
                    ">>> RECORDING STARTED"
                )

                print(
                    f"State: "
                    f"{current_state}"
                )

                print(
                    f"Eyes: "
                    f"{current_eye_state}"
                )

                print(
                    f"Gaze: "
                    f"{current_gaze_context}"
                )

                print(
                    f"Scene: "
                    f"{current_scene_context}"
                )

                print()

            else:

                if (
                    recording_start_time
                    is not None
                ):

                    total_recording_time += (
                        time.time()
                        -
                        recording_start_time
                    )

                recording_start_time = (
                    None
                )

                print()
                print(
                    ">>> RECORDING STOPPED"
                )

                print(
                    f"Frames: "
                    f"{total_frames}"
                )

                print(
                    f"Recording time: "
                    f"{format_duration(
                        total_recording_time
                    )}"
                )

                print()

    # ========================================================
    # FINALIZE RECORDING TIME
    # ========================================================

    if (
        recording
        and
        recording_start_time
        is not None
    ):

        total_recording_time += (
            time.time()
            -
            recording_start_time
        )

    # ========================================================
    # RELEASE RESOURCES
    # ========================================================

    driver_camera.release()

    scene_camera.release()

    csv_file.close()

    cv2.destroyAllWindows()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "MULTIMODAL DATASET COLLECTION COMPLETE"
    )
    print("=" * 70)
    print()

    print(
        f"Session:"
        f" {session_name}"
    )

    print(
        f"Total frames:"
        f" {total_frames}"
    )

    print(
        f"Recording time:"
        f" {format_duration(
            total_recording_time
        )}"
    )

    print()

    print(
        f"Driver images:"
        f" {driver_path}"
    )

    print(
        f"Scene images:"
        f" {scene_path}"
    )

    print(
        f"Annotations:"
        f" {csv_path}"
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()