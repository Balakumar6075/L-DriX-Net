import cv2
import numpy as np


class HeadPoseEstimator:
    """
    Lightweight geometric head-pose estimator.

    This implementation estimates approximate:

        yaw   -> left/right head orientation
        pitch -> up/down head orientation
        roll  -> head tilt

    It uses YuNet facial landmarks instead of solvePnP.

    Why?
    ----
    The live webcam is not necessarily calibrated and we do not
    have subject-specific 3D facial measurements. A generic PnP
    model can therefore produce orientation ambiguities such as
    a pitch near +/-180 degrees.

    This estimator uses 2D facial geometry to provide stable
    normalized head-orientation features for the live gaze pipeline.

    IMPORTANT:
    These values are approximate head-orientation features and
    should not be interpreted as calibrated physical angles.
    """

    def __init__(
        self,
        yaw_scale=55.0,
        pitch_scale=45.0,
    ):
        # ------------------------------------------------------
        # Sensitivity parameters
        # ------------------------------------------------------

        self.yaw_scale = float(yaw_scale)
        self.pitch_scale = float(pitch_scale)

    # ==========================================================
    # Utility Functions
    # ==========================================================

    @staticmethod
    def _point(point):
        """
        Convert a 2D point to a NumPy array.
        """

        return np.array(
            [
                float(point[0]),
                float(point[1]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _clip(value, minimum, maximum):
        """
        Clip a value to a specified range.
        """

        return float(
            np.clip(
                value,
                minimum,
                maximum,
            )
        )

    # ==========================================================
    # Head Pose Estimation
    # ==========================================================

    def estimate(
        self,
        face,
        image_width,
        image_height,
    ):
        """
        Estimate approximate head pose using facial geometry.

        Args:
            face:
                Dictionary returned by FaceDetector.

            image_width:
                Original image width.

            image_height:
                Original image height.

        Returns:
            Dictionary containing:

                yaw
                pitch
                roll

                yaw_normalized
                pitch_normalized
                roll_normalized

                face_center
                eye_center
                eye_vector
                eye_distance

        """

        if face is None:
            return None

        if image_width <= 0 or image_height <= 0:
            return None

        # ------------------------------------------------------
        # Read facial landmarks
        # ------------------------------------------------------

        right_eye = self._point(
            face["right_eye"]
        )

        left_eye = self._point(
            face["left_eye"]
        )

        nose = self._point(
            face["nose"]
        )

        right_mouth = self._point(
            face["right_mouth"]
        )

        left_mouth = self._point(
            face["left_mouth"]
        )

        # ------------------------------------------------------
        # Face dimensions
        # ------------------------------------------------------

        face_x = float(face["x"])
        face_y = float(face["y"])
        face_w = float(face["w"])
        face_h = float(face["h"])

        if face_w <= 0 or face_h <= 0:
            return None

        # ------------------------------------------------------
        # Eye center
        # ------------------------------------------------------

        eye_center = (
            right_eye + left_eye
        ) / 2.0

        # ------------------------------------------------------
        # Mouth center
        # ------------------------------------------------------

        mouth_center = (
            right_mouth + left_mouth
        ) / 2.0

        # ------------------------------------------------------
        # Eye distance
        # ------------------------------------------------------

        eye_vector = (
            left_eye - right_eye
        )

        eye_distance = float(
            np.linalg.norm(
                eye_vector
            )
        )

        if eye_distance < 1e-6:
            return None

        # ======================================================
        # 1. ROLL
        # ======================================================
        #
        # The line joining the two eyes should be approximately
        # horizontal when the head is upright.
        #
        # atan2(dy, dx) gives the eye-line tilt.
        # ======================================================

        roll_radians = np.arctan2(
            eye_vector[1],
            eye_vector[0],
        )

        roll = float(
            np.degrees(
                roll_radians
            )
        )

        # Keep roll within a useful range.
        if roll > 90.0:
            roll -= 180.0

        if roll < -90.0:
            roll += 180.0

        roll = self._clip(
            roll,
            -45.0,
            45.0,
        )

        # ======================================================
        # 2. YAW
        # ======================================================
        #
        # Compare the nose position with the midpoint between
        # the two eyes.
        #
        # If the nose moves horizontally relative to the eye
        # center, the head is likely rotated left/right.
        #
        # Normalize by eye distance so that the measurement
        # remains approximately scale-independent.
        # ======================================================

        nose_horizontal_offset = (
            nose[0] - eye_center[0]
        )

        yaw_normalized = (
            nose_horizontal_offset
            / eye_distance
        )

        # Convert normalized displacement into an approximate
        # angle.
        yaw = float(
            np.degrees(
                np.arctan(
                    yaw_normalized
                    * 2.0
                )
            )
        )

        yaw *= (
            self.yaw_scale
            / 55.0
        )

        yaw = self._clip(
            yaw,
            -60.0,
            60.0,
        )

        # ======================================================
        # 3. PITCH
        # ======================================================
        #
        # Compare the nose position relative to the eye-to-mouth
        # vertical geometry.
        #
        # For an approximately frontal neutral face:
        #
        #       eyes
        #         |
        #       nose
        #         |
        #       mouth
        #
        # We use the relative position rather than absolute
        # pixels.
        # ======================================================

        eye_to_mouth_vector = (
            mouth_center - eye_center
        )

        eye_to_mouth_distance = float(
            np.linalg.norm(
                eye_to_mouth_vector
            )
        )

        if eye_to_mouth_distance < 1e-6:
            return None

        nose_from_eye = (
            nose - eye_center
        )

        # Projection of the nose along the eye-to-mouth direction.
        mouth_direction = (
            eye_to_mouth_vector
            / eye_to_mouth_distance
        )

        nose_projection = float(
            np.dot(
                nose_from_eye,
                mouth_direction,
            )
        )

        # Expected nose position as a fraction of the
        # eye-to-mouth distance.
        #
        # Approximately half-way is used as the neutral
        # reference.
        expected_nose_projection = (
            eye_to_mouth_distance * 0.50
        )

        pitch_error = (
            nose_projection
            - expected_nose_projection
        )

        pitch_normalized = (
            pitch_error
            / eye_distance
        )

        pitch = float(
            np.degrees(
                np.arctan(
                    pitch_normalized
                    * 2.0
                )
            )
        )

        pitch *= (
            self.pitch_scale
            / 45.0
        )

        pitch = self._clip(
            pitch,
            -45.0,
            45.0,
        )

        # ======================================================
        # Normalized face coordinates
        # ======================================================

        face_center = np.array(
            [
                face_x + face_w / 2.0,
                face_y + face_h / 2.0,
            ],
            dtype=np.float32,
        )

        eye_center_normalized = (
            eye_center
            - np.array(
                [
                    face_x,
                    face_y,
                ],
                dtype=np.float32,
            )
        ) / np.array(
            [
                face_w,
                face_h,
            ],
            dtype=np.float32,
        )

        nose_normalized = (
            nose
            - np.array(
                [
                    face_x,
                    face_y,
                ],
                dtype=np.float32,
            )
        ) / np.array(
            [
                face_w,
                face_h,
            ],
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Final normalized pose values
        # ------------------------------------------------------

        yaw_normalized = self._clip(
            yaw / 60.0,
            -1.0,
            1.0,
        )

        pitch_normalized = self._clip(
            pitch / 45.0,
            -1.0,
            1.0,
        )

        roll_normalized = self._clip(
            roll / 45.0,
            -1.0,
            1.0,
        )

        return {
            # Approximate angles
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,

            # Normalized values
            "yaw_normalized": yaw_normalized,
            "pitch_normalized": pitch_normalized,
            "roll_normalized": roll_normalized,

            # Geometry
            "face_center": (
                face_center.tolist()
            ),

            "eye_center": (
                eye_center.tolist()
            ),

            "eye_vector": (
                eye_vector.tolist()
            ),

            "eye_distance": eye_distance,

            "eye_center_normalized": (
                eye_center_normalized.tolist()
            ),

            "nose_normalized": (
                nose_normalized.tolist()
            ),

            "mouth_center": (
                mouth_center.tolist()
            ),

            "eye_to_mouth_distance": (
                eye_to_mouth_distance
            ),
        }

    # ==========================================================
    # Draw Head Pose
    # ==========================================================

    @staticmethod
    def draw_axes(
        frame,
        face,
        head_pose,
        axis_length=None,
    ):
        """
        Draw a 2D visualization of estimated head orientation.

        Unlike the previous implementation, this does not pretend
        to be a calibrated 3D coordinate system.
        """

        if frame is None:
            return frame

        if face is None:
            return frame

        if head_pose is None:
            return frame

        output = frame.copy()

        # ------------------------------------------------------
        # Nose origin
        # ------------------------------------------------------

        nose = (
            int(round(face["nose"][0])),
            int(round(face["nose"][1])),
        )

        # ------------------------------------------------------
        # Axis length
        # ------------------------------------------------------

        if axis_length is None:
            axis_length = max(
                50,
                int(face["w"] * 0.55),
            )

        axis_length = float(
            axis_length
        )

        # ------------------------------------------------------
        # Angles
        # ------------------------------------------------------

        yaw = np.radians(
            head_pose["yaw"]
        )

        pitch = np.radians(
            head_pose["pitch"]
        )

        roll = np.radians(
            head_pose["roll"]
        )

        # ------------------------------------------------------
        # Horizontal yaw direction
        # ------------------------------------------------------

        yaw_x = (
            np.sin(yaw)
            * axis_length
        )

        yaw_y = 0.0

        # ------------------------------------------------------
        # Vertical pitch direction
        # ------------------------------------------------------

        pitch_x = 0.0

        pitch_y = (
            -np.sin(pitch)
            * axis_length
        )

        # ------------------------------------------------------
        # Roll direction
        # ------------------------------------------------------

        roll_x = (
            np.cos(roll)
            * axis_length
        )

        roll_y = (
            np.sin(roll)
            * axis_length
        )

        # ------------------------------------------------------
        # Draw yaw axis
        # ------------------------------------------------------

        yaw_end = (
            int(round(nose[0] + yaw_x)),
            int(round(nose[1] + yaw_y)),
        )

        cv2.arrowedLine(
            output,
            nose,
            yaw_end,
            (0, 0, 255),
            3,
            tipLength=0.15,
        )

        # ------------------------------------------------------
        # Draw pitch axis
        # ------------------------------------------------------

        pitch_end = (
            int(round(nose[0] + pitch_x)),
            int(round(nose[1] + pitch_y)),
        )

        cv2.arrowedLine(
            output,
            nose,
            pitch_end,
            (0, 255, 0),
            3,
            tipLength=0.15,
        )

        # ------------------------------------------------------
        # Draw roll axis
        # ------------------------------------------------------

        roll_end = (
            int(round(nose[0] + roll_x)),
            int(round(nose[1] + roll_y)),
        )

        cv2.arrowedLine(
            output,
            nose,
            roll_end,
            (255, 0, 0),
            3,
            tipLength=0.15,
        )

        # ------------------------------------------------------
        # Draw nose origin
        # ------------------------------------------------------

        cv2.circle(
            output,
            nose,
            6,
            (255, 255, 255),
            -1,
        )

        return output

    # ==========================================================
    # Draw Pose Information
    # ==========================================================

    @staticmethod
    def draw_info(
        frame,
        head_pose,
        position=(20, 30),
    ):
        """
        Draw numerical head-pose information.
        """

        if frame is None:
            return frame

        if head_pose is None:
            return frame

        output = frame.copy()

        x, y = position

        lines = [
            f"Yaw:   {head_pose['yaw']:.2f} deg",
            f"Pitch: {head_pose['pitch']:.2f} deg",
            f"Roll:  {head_pose['roll']:.2f} deg",
        ]

        for index, text in enumerate(lines):

            text_y = (
                y
                + index * 28
            )

            cv2.putText(
                output,
                text,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output


# ==============================================================
# Standalone Test
# ==============================================================

if __name__ == "__main__":

    print(
        "HeadPoseEstimator initialized successfully."
    )

    estimator = HeadPoseEstimator()

    print(
        "Estimator type:"
    )

    print(
        "2D facial geometry"
    )

    print(
        "\nOutputs:"
    )

    print(
        "  yaw"
    )

    print(
        "  pitch"
    )

    print(
        "  roll"
    )

    print(
        "\nThis estimator does not use solvePnP."
    )