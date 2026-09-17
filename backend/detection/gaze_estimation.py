import numpy as np


class GazeEstimator:
    """
    Estimate a live 3D gaze representation from:

        1. Eye geometry
        2. Head pose

    The output is converted into the 24-dimensional representation
    expected by the trained L-DriX-Net model.

    IMPORTANT
    ---------
    The original dataset contains calibrated ground-truth:

        Gaze_Loc_2D
        Gaze_Loc_3D
        Left_Gaze_Dir
        Right_Gaze_Dir

    A normal webcam cannot directly measure the first two values.

    Therefore this class creates an APPROXIMATE live representation.
    It is intended for live inference and not as a replacement for
    the dataset's calibrated ground-truth annotations.
    """

    def __init__(
        self,
        scene_width=942,
        scene_height=489,
    ):
        """
        Args:
            scene_width:
                Expected scene-camera width.

            scene_height:
                Expected scene-camera height.
        """

        self.scene_width = float(scene_width)
        self.scene_height = float(scene_height)

    # ==========================================================
    # Utility Functions
    # ==========================================================

    @staticmethod
    def _normalize_vector(vector):
        """
        Normalize a vector safely.
        """

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        magnitude = np.linalg.norm(
            vector
        )

        if magnitude < 1e-8:
            return np.zeros_like(
                vector
            )

        return vector / magnitude

    @staticmethod
    def _clip(value, minimum, maximum):
        return float(
            np.clip(
                value,
                minimum,
                maximum,
            )
        )

    # ==========================================================
    # Estimate Gaze Direction
    # ==========================================================

    def _estimate_gaze_direction(
        self,
        yaw,
        pitch,
    ):
        """
        Convert head orientation into an approximate forward
        gaze direction.

        Coordinate convention:

            X -> horizontal
            Y -> vertical
            Z -> forward

        The resulting vector is normalized.
        """

        yaw_rad = np.radians(
            float(yaw)
        )

        pitch_rad = np.radians(
            float(pitch)
        )

        x = np.sin(
            yaw_rad
        )

        y = -np.sin(
            pitch_rad
        )

        z = (
            np.cos(yaw_rad)
            * np.cos(pitch_rad)
        )

        direction = np.array(
            [
                x,
                y,
                z,
            ],
            dtype=np.float32,
        )

        return self._normalize_vector(
            direction
        )

    # ==========================================================
    # Estimate Eye Directions
    # ==========================================================

    def _estimate_eye_directions(
        self,
        eye_features,
        head_pose,
    ):
        """
        Estimate left and right eye gaze directions.

        At this stage we use head orientation plus small
        eye-position corrections.

        This is intentionally conservative because YuNet gives
        eye-center landmarks, not pupil landmarks.
        """

        yaw = float(
            head_pose["yaw"]
        )

        pitch = float(
            head_pose["pitch"]
        )

        # ------------------------------------------------------
        # Base gaze direction from head orientation
        # ------------------------------------------------------

        base_direction = (
            self._estimate_gaze_direction(
                yaw,
                pitch,
            )
        )

        # ------------------------------------------------------
        # Eye-center horizontal displacement
        # ------------------------------------------------------

        eye_center_offset = np.asarray(
            eye_features[
                "eye_center_offset_normalized"
            ],
            dtype=np.float32,
        )

        horizontal_offset = float(
            eye_center_offset[0]
        )

        vertical_offset = float(
            eye_center_offset[1]
        )

        # ------------------------------------------------------
        # Small eye contribution
        #
        # We deliberately keep this contribution limited.
        # ------------------------------------------------------

        eye_yaw_correction = (
            horizontal_offset * 0.35
        )

        eye_pitch_correction = (
            vertical_offset * 0.20
        )

        # ------------------------------------------------------
        # Left eye
        # ------------------------------------------------------

        left_yaw = (
            yaw
            + eye_yaw_correction * 20.0
        )

        left_pitch = (
            pitch
            - eye_pitch_correction * 20.0
        )

        left_direction = (
            self._estimate_gaze_direction(
                left_yaw,
                left_pitch,
            )
        )

        # ------------------------------------------------------
        # Right eye
        # ------------------------------------------------------

        right_yaw = (
            yaw
            + eye_yaw_correction * 20.0
        )

        right_pitch = (
            pitch
            - eye_pitch_correction * 20.0
        )

        right_direction = (
            self._estimate_gaze_direction(
                right_yaw,
                right_pitch,
            )
        )

        return (
            left_direction,
            right_direction,
        )

    # ==========================================================
    # Estimate 2D Scene Gaze Location
    # ==========================================================

    def _estimate_gaze_location_2d(
        self,
        gaze_direction,
    ):
        """
        Convert normalized gaze direction into an approximate
        scene-camera location.

        This uses the scene dimensions supplied to the class.

        The result is:

            [x, y]

        in scene-image coordinates.
        """

        direction = (
            self._normalize_vector(
                gaze_direction
            )
        )

        x_direction = float(
            direction[0]
        )

        y_direction = float(
            direction[1]
        )

        # ------------------------------------------------------
        # Map direction range to image range.
        #
        # x = -1 -> left
        # x = +1 -> right
        #
        # y = +1 -> upward
        # y = -1 -> downward
        # ------------------------------------------------------

        x_normalized = (
            0.5
            + 0.5 * x_direction
        )

        y_normalized = (
            0.5
            - 0.5 * y_direction
        )

        x_normalized = self._clip(
            x_normalized,
            0.0,
            1.0,
        )

        y_normalized = self._clip(
            y_normalized,
            0.0,
            1.0,
        )

        x = (
            x_normalized
            * self.scene_width
        )

        y = (
            y_normalized
            * self.scene_height
        )

        return np.array(
            [
                x,
                y,
            ],
            dtype=np.float32,
        )

    # ==========================================================
    # Estimate 3D Gaze Location
    # ==========================================================

    def _estimate_gaze_location_3d(
        self,
        gaze_direction,
    ):
        """
        Create an approximate 3D gaze point.

        We use a fixed reference depth because a monocular
        webcam does not provide reliable absolute target depth.
        """

        direction = (
            self._normalize_vector(
                gaze_direction
            )
        )

        reference_depth = 1.0

        point = (
            direction
            * reference_depth
        )

        return point.astype(
            np.float32
        )

    # ==========================================================
    # Build 24-D Vector
    # ==========================================================

    def build_vector(
        self,
        eye_features,
        head_pose,
    ):
        """
        Build the complete 24-dimensional gaze input.

        Layout:

            [0:2]    Gaze_Loc_2D
            [2:5]    Gaze_Loc_3D
            [5:8]    Left_Gaze_Dir
            [8:11]   Right_Gaze_Dir
            [11:24]  zero padding

        Returns:
            NumPy array with shape (24,)
        """

        if eye_features is None:
            raise ValueError(
                "eye_features cannot be None"
            )

        if head_pose is None:
            raise ValueError(
                "head_pose cannot be None"
            )

        # ------------------------------------------------------
        # Eye directions
        # ------------------------------------------------------

        (
            left_direction,
            right_direction,
        ) = self._estimate_eye_directions(
            eye_features,
            head_pose,
        )

        # ------------------------------------------------------
        # Average gaze direction
        # ------------------------------------------------------

        average_direction = (
            left_direction
            + right_direction
        ) / 2.0

        average_direction = (
            self._normalize_vector(
                average_direction
            )
        )

        # ------------------------------------------------------
        # 2D gaze location
        # ------------------------------------------------------

        gaze_location_2d = (
            self._estimate_gaze_location_2d(
                average_direction
            )
        )

        # ------------------------------------------------------
        # 3D gaze location
        # ------------------------------------------------------

        gaze_location_3d = (
            self._estimate_gaze_location_3d(
                average_direction
            )
        )

        # ------------------------------------------------------
        # Build first 11 values
        # ------------------------------------------------------

        vector = np.concatenate(
            [
                gaze_location_2d,
                gaze_location_3d,
                left_direction,
                right_direction,
            ]
        ).astype(
            np.float32
        )

        # ------------------------------------------------------
        # Ensure exactly 24 dimensions
        # ------------------------------------------------------

        if vector.shape[0] < 24:

            padding = np.zeros(
                24 - vector.shape[0],
                dtype=np.float32,
            )

            vector = np.concatenate(
                [
                    vector,
                    padding,
                ]
            )

        elif vector.shape[0] > 24:

            vector = vector[:24]

        return vector

    # ==========================================================
    # Full Estimation
    # ==========================================================

    def estimate(
        self,
        eye_features,
        head_pose,
    ):
        """
        Estimate all gaze information and return both the
        structured representation and the 24-D model vector.
        """

        (
            left_direction,
            right_direction,
        ) = self._estimate_eye_directions(
            eye_features,
            head_pose,
        )

        average_direction = (
            left_direction
            + right_direction
        ) / 2.0

        average_direction = (
            self._normalize_vector(
                average_direction
            )
        )

        gaze_location_2d = (
            self._estimate_gaze_location_2d(
                average_direction
            )
        )

        gaze_location_3d = (
            self._estimate_gaze_location_3d(
                average_direction
            )
        )

        vector = np.concatenate(
            [
                gaze_location_2d,
                gaze_location_3d,
                left_direction,
                right_direction,
                np.zeros(
                    13,
                    dtype=np.float32,
                ),
            ]
        )

        vector = vector[:24].astype(
            np.float32
        )

        return {
            "gaze_location_2d": (
                gaze_location_2d.tolist()
            ),

            "gaze_location_3d": (
                gaze_location_3d.tolist()
            ),

            "left_gaze_direction": (
                left_direction.tolist()
            ),

            "right_gaze_direction": (
                right_direction.tolist()
            ),

            "average_gaze_direction": (
                average_direction.tolist()
            ),

            "vector_24d": (
                vector.tolist()
            ),
        }


# ==============================================================
# Standalone Test
# ==============================================================

if __name__ == "__main__":

    print(
        "GazeEstimator initialized successfully."
    )

    estimator = GazeEstimator()

    # Example neutral head pose
    head_pose = {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
    }

    # Example eye geometry
    eye_features = {
        "eye_center_offset_normalized": [
            0.0,
            0.0,
        ]
    }

    result = estimator.estimate(
        eye_features,
        head_pose,
    )

    print(
        "\nEstimated gaze:"
    )

    print(
        "2D location:",
        result["gaze_location_2d"],
    )

    print(
        "3D location:",
        result["gaze_location_3d"],
    )

    print(
        "Left direction:",
        result["left_gaze_direction"],
    )

    print(
        "Right direction:",
        result["right_gaze_direction"],
    )

    print(
        "Average direction:",
        result["average_gaze_direction"],
    )

    print(
        "\n24-D vector:"
    )

    print(
        result["vector_24d"]
    )

    print(
        "\nVector length:",
        len(result["vector_24d"]),
    )