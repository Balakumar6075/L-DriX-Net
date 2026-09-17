import cv2
import numpy as np


class EyeDetector:
    """
    Extracts eye-related geometry from YuNet facial landmarks.

    YuNet provides approximate eye-center locations. This class
    converts those locations into normalized features relative
    to the detected face.
    """

    def __init__(self):
        pass

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

    def extract(self, face):
        """
        Extract normalized eye geometry.

        Args:
            face: dictionary returned by FaceDetector.

        Returns:
            dictionary containing eye positions and normalized
            geometric features.
        """

        if face is None:
            return None

        right_eye = self._point(face["right_eye"])
        left_eye = self._point(face["left_eye"])
        nose = self._point(face["nose"])

        face_x = float(face["x"])
        face_y = float(face["y"])
        face_w = float(face["w"])
        face_h = float(face["h"])

        if face_w <= 0 or face_h <= 0:
            return None

        # Normalize eye coordinates to the detected face box.
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

        # Eye separation normalized by face width.
        eye_distance = self._distance(
            right_eye,
            left_eye,
        )

        eye_distance_norm = eye_distance / face_w

        # Midpoint between the two eyes.
        eye_center = (right_eye + left_eye) / 2.0

        eye_center_norm = np.array(
            [
                (eye_center[0] - face_x) / face_w,
                (eye_center[1] - face_y) / face_h,
            ],
            dtype=np.float32,
        )

        # Horizontal asymmetry of the eyes around the face center.
        face_center = np.array(
            [
                face_x + face_w / 2.0,
                face_y + face_h / 2.0,
            ],
            dtype=np.float32,
        )

        eye_center_offset = eye_center - face_center

        eye_center_offset_norm = eye_center_offset / np.array(
            [face_w, face_h],
            dtype=np.float32,
        )

        return {
            "right_eye": right_eye.tolist(),
            "left_eye": left_eye.tolist(),
            "nose": nose.tolist(),

            "right_eye_normalized": right_eye_norm.tolist(),
            "left_eye_normalized": left_eye_norm.tolist(),
            "nose_normalized": nose_norm.tolist(),

            "eye_center": eye_center.tolist(),
            "eye_center_normalized": eye_center_norm.tolist(),

            "eye_distance": eye_distance,
            "eye_distance_normalized": float(
                eye_distance_norm
            ),

            "eye_center_offset": eye_center_offset.tolist(),
            "eye_center_offset_normalized": (
                eye_center_offset_norm.tolist()
            ),
        }

    @staticmethod
    def draw_features(frame, eye_features):
        """
        Draw eye and nose landmarks on an image.
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

        return output


if __name__ == "__main__":
    print("EyeDetector module initialized successfully.")