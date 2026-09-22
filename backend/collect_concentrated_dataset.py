import cv2
import csv
import os
import time


# ============================================================
# L-DriX-Net
# CONCENTRATED SESSION DATASET COLLECTOR
# ============================================================
#
# IMPORTANT:
# This collector is ONLY for the CONCENTRATED session.
#
# For this session:
#
#   DRIVER CAMERA  -> USED
#   SCENE CAMERA   -> NOT USED
#
# Every recorded frame is automatically labeled:
#
#   CONCENTRATED
#
# The scene camera is intentionally ignored because the physical
# scene camera will be positioned toward the right-side road.
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
# ONLY DRIVER CAMERA IS USED
# ------------------------------------------------------------

CAMERA_DRIVER = 0


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


CAMERA_FPS = 30


JPEG_QUALITY = 95


# ============================================================
# SESSION PREFIX
# ============================================================

SESSION_PREFIX = "concentrated_session_"


# ============================================================
# DATASET LABEL
# ============================================================

STATE = "CONCENTRATED"


# ============================================================
# CREATE NEW SESSION
# ============================================================

def create_session():

    # --------------------------------------------------------
    # Create main dataset directory
    # --------------------------------------------------------

    os.makedirs(
        DATASET_ROOT,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Find existing concentrated sessions
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Determine next session number
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Session name
    # --------------------------------------------------------

    session_name = (
        f"{SESSION_PREFIX}"
        f"{next_number:03d}"
    )


    # --------------------------------------------------------
    # Session directory
    # --------------------------------------------------------

    session_path = os.path.join(
        DATASET_ROOT,
        session_name
    )


    # --------------------------------------------------------
    # Driver image directory
    # --------------------------------------------------------

    driver_path = os.path.join(
        session_path,
        "driver"
    )


    os.makedirs(
        driver_path,
        exist_ok=True
    )


    return (
        session_name,
        session_path,
        driver_path
    )


# ============================================================
# OPEN DRIVER CAMERA
# ============================================================

def open_driver_camera():

    camera = cv2.VideoCapture(
        CAMERA_DRIVER,
        cv2.CAP_DSHOW
    )


    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )


    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )


    return camera


# ============================================================
# CREATE ANNOTATION CSV
# ============================================================

def create_annotation_csv(
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


    # --------------------------------------------------------
    # CSV HEADER
    # --------------------------------------------------------

    csv_writer.writerow([
        "frame_id",
        "timestamp",
        "state"
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
        "L-DriX-Net CONCENTRATED SESSION "
        "DATASET COLLECTOR"
    )
    print("=" * 70)
    print()


    print(
        "MODE: CONCENTRATED"
    )


    print()


    print(
        "Camera:"
    )


    print(
        "  Driver camera : Camera 0"
    )


    print(
        "  Scene camera  : DISABLED"
    )


    print()


    print(
        "Every recorded frame will be labeled:"
    )


    print(
        "  CONCENTRATED"
    )


    print()


    print(
        "Controls:"
    )


    print(
        "  SPACE = Start / Stop recording"
    )


    print(
        "  Q     = Quit"
    )


    print()


    # ========================================================
    # CREATE SESSION
    # ========================================================

    (
        session_name,
        session_path,
        driver_path
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
    ) = create_annotation_csv(
        csv_path
    )


    # ========================================================
    # OPEN CAMERA
    # ========================================================

    camera = open_driver_camera()


    if not camera.isOpened():

        print()
        print(
            "ERROR: Could not open "
            "driver camera."
        )


        csv_file.close()


        return


    print(
        "Driver camera opened successfully."
    )


    print()


    # ========================================================
    # RECORDING STATE
    # ========================================================

    recording = False


    frame_id = 0


    total_frames = 0


    recording_start_time = None


    total_recording_time = 0.0


    # ========================================================
    # FPS CALCULATION
    # ========================================================

    fps_counter = 0


    fps_start = time.time()


    display_fps = 0.0


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # READ CAMERA FRAME
        # ----------------------------------------------------

        success, frame = (
            camera.read()
        )


        if not success:

            print(
                "ERROR: Could not read "
                "camera frame."
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
        # CREATE DISPLAY FRAME
        # ====================================================

        display = frame.copy()


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
        # CURRENT RECORDING DURATION
        # ====================================================

        current_recording_duration = (
            total_recording_time
        )


        if (
            recording
            and
            recording_start_time
            is not None
        ):

            current_recording_duration += (
                time.time()
                -
                recording_start_time
            )


        # ====================================================
        # DRAW INFORMATION
        # ====================================================

        cv2.putText(
            display,
            "L-DriX-Net",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            "SESSION: CONCENTRATED",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display,
            f"STATUS: {status_text}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )


        cv2.putText(
            display,
            f"FRAMES: {total_frames}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            f"FPS: {display_fps:.1f}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            (
                "TIME: "
                +
                format_duration(
                    current_recording_duration
                )
            ),
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            display,
            "SPACE = START / STOP",
            (
                20,
                FRAME_HEIGHT - 60
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            "Q = QUIT",
            (
                20,
                FRAME_HEIGHT - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # RECORD CURRENT FRAME
        # ====================================================

        if recording:

            # ------------------------------------------------
            # Filename
            # ------------------------------------------------

            filename = (
                f"{frame_id:06d}.jpg"
            )


            # ------------------------------------------------
            # Full driver image path
            # ------------------------------------------------

            driver_file = os.path.join(
                driver_path,
                filename
            )


            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            cv2.imwrite(
                driver_file,
                frame,
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
                STATE
            ])


            csv_file.flush()


            # ------------------------------------------------
            # Update counters
            # ------------------------------------------------

            frame_id += 1


            total_frames += 1


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "L-DriX-Net "
            "Concentrated Dataset Collector",
            display
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
        # START / STOP RECORDING
        # ====================================================

        elif key == 32:

            recording = not recording


            if recording:

                # --------------------------------------------
                # START RECORDING
                # --------------------------------------------

                recording_start_time = (
                    time.time()
                )


                print()
                print(
                    ">>> RECORDING STARTED"
                )


                print(
                    "State: CONCENTRATED"
                )


                print()


            else:

                # --------------------------------------------
                # STOP RECORDING
                # --------------------------------------------

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
    # FINALIZE CURRENT RECORDING
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

    camera.release()


    csv_file.close()


    cv2.destroyAllWindows()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "CONCENTRATED DATASET COLLECTION COMPLETE"
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
        f"Annotations:"
        f" {csv_path}"
    )


    print()


    print(
        "Label:"
    )


    print(
        "  CONCENTRATED"
    )


    print()


    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()