import cv2
import numpy as np


class EyeDetector:
    """
    Eye feature extractor using YuNet eye-center landmarks.

    In addition to the original eye geometry, this version performs
    lightweight eye-region analysis to estimate whether each eye
    appears open or closed.

    NOTE:
        This is a prototype eye-state estimator.
        It is not a medically validated eye-closure detector.
    """

    def __init__(self):
        # Size of the crop around each YuNet eye-center point.
        self.eye_crop_width_ratio = 0.28
        self.eye_crop_height_ratio = 0.16

        # Threshold for the eye-region darkness ratio.
        #
        # This is intentionally conservative and should be tuned
        # using live camera samples.
        self.dark_ratio_threshold = 0.34

    @staticmethod
    def _point(point):
        return np.array(
            [float(point[0]), float(point[1])],
            dtype=np.float32,
        )

    @staticmethod
    def _distance(point_a, point_b):
        return float(
            np.linalg.norm(
                EyeDetector._point(point_a)
                - EyeDetector._point(point_b)
            )
        )

    def _crop_eye(
        self,
        frame,
        eye_point,
        face_w,
        face_h,
    ):
        """
        Crop a small region around the detected eye center.
        """

        if frame is None:
            return None

        x = int(round(float(eye_point[0])))
        y = int(round(float(eye_point[1])))

        crop_w = max(
            10,
            int(face_w * self.eye_crop_width_ratio)
        )

        crop_h = max(
            8,
            int(face_h * self.eye_crop_height_ratio)
        )

        x1 = max(0, x - crop_w // 2)
        y1 = max(0, y - crop_h // 2)

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

        return frame[y1:y2, x1:x2]

    def _analyze_eye_region(self, eye_crop):
        """
        Estimate eye openness from the cropped eye region.

        Uses grayscale intensity distribution.

        This is only a lightweight heuristic and should not be
        interpreted as ground-truth eye-state detection.
        """

        if eye_crop is None or eye_crop.size == 0:
            return {
                "valid": False,
                "dark_ratio": 0.0,
                "mean_intensity": 0.0,
                "eye_open": False,
            }

        gray = cv2.cvtColor(
            eye_crop,
            cv2.COLOR_BGR2GRAY
        )

        # Small blur reduces camera noise.
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        mean_intensity = float(
            np.mean(gray)
        )

        # Pixels darker than this adaptive threshold.
        threshold = max(
            30.0,
            mean_intensity * 0.65
        )

        dark_ratio = float(
            np.mean(gray < threshold)
        )

        # Open eyes generally contain stronger local contrast
        # from iris/pupil/eye whites. Closed eyes tend to produce
        # a more compressed region.
        local_std = float(
            np.std(gray)
        )

        eye_open = (
            local_std > 18.0
            and dark_ratio < self.dark_ratio_threshold
        )

        return {
            "valid": True,
            "dark_ratio": round(dark_ratio, 4),
            "mean_intensity": round(mean_intensity, 2),
            "local_std": round(local_std, 2),
            "eye_open": bool(eye_open),
        }

    def extract(self, face, frame=None):
        """
        Extract normalized eye geometry and estimated eye state.

        Args:
            face:
                Dictionary returned by FaceDetector.

            frame:
                Original BGR camera frame.

        Returns:
            Dictionary containing eye geometry and eye-state features.
        """

        if face is None:
            return None

        required_keys = [
            "right_eye",
            "left_eye",
            "nose",
            "x",
            "y",
            "w",
            "h",
        ]

        for key in required_keys:
            if key not in face:
                return None

        right_eye = self._point(
            face["right_eye"]
        )

        left_eye = self._point(
            face["left_eye"]
        )

        nose = self._point(
            face["nose"]
        )

        face_x = float(face["x"])
        face_y = float(face["y"])
        face_w = float(face["w"])
        face_h = float(face["h"])

        if face_w <= 0 or face_h <= 0:
            return None

        # ---------------------------------------------------------
        # NORMALIZED EYE GEOMETRY
        # ---------------------------------------------------------

        right_eye_norm = np.array(
            [
                (right_eye[0] - face_x) / face_w,
                (right_eye[1] - face_y) / face_h,
            ],
            dtype=np.float32,
        )

        left_eye_norm = np.array(
            [
                (left_eye[0] - face_x) / face_w,
                (left_eye[1] - face_y) / face_h,
            ],
            dtype=np.float32,
        )

        nose_norm = np.array(
            [
                (nose[0] - face_x) / face_w,
                (nose[1] - face_y) / face_h,
            ],
            dtype=np.float32,
        )

        # Eye separation.
        eye_distance = self._distance(
            right_eye,
            left_eye
        )

        eye_distance_norm = (
            eye_distance / face_w
        )

        # Eye midpoint.
        eye_center = (
            right_eye + left_eye
        ) / 2.0

        eye_center_norm = np.array(
            [
                (eye_center[0] - face_x) / face_w,
                (eye_center[1] - face_y) / face_h,
            ],
            dtype=np.float32,
        )

        # Face center.
        face_center = np.array(
            [
                face_x + face_w / 2.0,
                face_y + face_h / 2.0,
            ],
            dtype=np.float32,
        )

        eye_center_offset = (
            eye_center - face_center
        )

        eye_center_offset_norm = (
            eye_center_offset
            / np.array(
                [face_w, face_h],
                dtype=np.float32,
            )
        )

        # ---------------------------------------------------------
        # EYE REGION ANALYSIS
        # ---------------------------------------------------------

        right_crop = self._crop_eye(
            frame,
            right_eye,
            face_w,
            face_h,
        )

        left_crop = self._crop_eye(
            frame,
            left_eye,
            face_w,
            face_h,
        )

        right_state = self._analyze_eye_region(
            right_crop
        )

        left_state = self._analyze_eye_region(
            left_crop
        )

        # Both eyes must appear closed before we classify the
        # driver's eyes as closed.
        both_valid = (
            right_state["valid"]
            and left_state["valid"]
        )

        if both_valid:
            eyes_closed = (
                not right_state["eye_open"]
                and not left_state["eye_open"]
            )
        else:
            eyes_closed = False

        return {
            # Original geometry
            "right_eye": right_eye.tolist(),
            "left_eye": left_eye.tolist(),
            "nose": nose.tolist(),

            "right_eye_normalized":
                right_eye_norm.tolist(),

            "left_eye_normalized":
                left_eye_norm.tolist(),

            "nose_normalized":
                nose_norm.tolist(),

            "eye_center":
                eye_center.tolist(),

            "eye_center_normalized":
                eye_center_norm.tolist(),

            "eye_distance":
                eye_distance,

            "eye_distance_normalized":
                float(eye_distance_norm),

            "eye_center_offset":
                eye_center_offset.tolist(),

            "eye_center_offset_normalized":
                eye_center_offset_norm.tolist(),

            # New eye-state information
            "right_eye_open":
                right_state["eye_open"],

            "left_eye_open":
                left_state["eye_open"],

            "eyes_closed":
                bool(eyes_closed),

            "right_eye_analysis":
                right_state,

            "left_eye_analysis":
                left_state,
        }

    @staticmethod
    def draw_features(frame, eye_features):
        """
        Draw eye landmarks and estimated eye state.
        """

        if frame is None or eye_features is None:
            return frame

        output = frame.copy()

        points = [
            (
                "right_eye",
                eye_features["right_eye"],
                (255, 0, 0),
            ),
            (
                "left_eye",
                eye_features["left_eye"],
                (0, 0, 255),
            ),
            (
                "nose",
                eye_features["nose"],
                (0, 255, 0),
            ),
            (
                "eye_center",
                eye_features["eye_center"],
                (255, 255, 0),
            ),
        ]

        for _, point, color in points:

            x = int(round(point[0]))
            y = int(round(point[1]))

            cv2.circle(
                output,
                (x, y),
                5,
                color,
                -1,
            )

        # ---------------------------------------------------------
        # EYE STATE TEXT
        # ---------------------------------------------------------

        eyes_closed = eye_features.get(
            "eyes_closed",
            False
        )

        if eyes_closed:
            state_text = "EYES CLOSED"
        else:
            state_text = "EYES OPEN"

        cv2.putText(
            output,
            state_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output


if __name__ == "__main__":

    print("=" * 60)
    print("L-DriX-Net Eye Detector")
    print("=" * 60)
    print("Eye geometry: READY")
    print("Eye-state estimation: READY")
    print("=" * 60)