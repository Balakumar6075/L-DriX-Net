import cv2
import numpy as np
from pathlib import Path


class FaceDetector:
    """
    YuNet-based face detector.

    OpenCV 5 compatible implementation using FaceDetectorYN.
    """

    def __init__(
        self,
        model_path=None,
        confidence_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    ):
        if model_path is None:
            project_root = Path(__file__).resolve().parents[2]

            model_path = (
                project_root
                / "backend"
                / "models"
                / "face_detection_yunet_2026may.onnx"
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found:\n{self.model_path}"
            )

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=(320, 320),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

        if self.detector is None:
            raise RuntimeError("Unable to create YuNet face detector.")

    def detect(self, frame):
        """
        Detect all faces in a BGR OpenCV image.

        Returns a list of dictionaries.

        Each dictionary contains:

            x
            y
            w
            h
            confidence

            right_eye
            left_eye
            nose
            right_mouth
            left_mouth
        """

        if frame is None:
            return []

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")

        if frame.size == 0:
            return []

        height, width = frame.shape[:2]

        if height <= 0 or width <= 0:
            return []

        # OpenCV 5 YuNet supports changing the input size dynamically.
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        results = []

        for face in faces:
            face = np.asarray(face, dtype=np.float32)

            if face.shape[0] < 15:
                continue

            x = float(face[0])
            y = float(face[1])
            w = float(face[2])
            h = float(face[3])

            confidence = float(face[14])

            result = {
                "x": int(round(x)),
                "y": int(round(y)),
                "w": int(round(w)),
                "h": int(round(h)),
                "confidence": confidence,

                # YuNet facial landmarks
                "right_eye": (
                    float(face[4]),
                    float(face[5]),
                ),

                "left_eye": (
                    float(face[6]),
                    float(face[7]),
                ),

                "nose": (
                    float(face[8]),
                    float(face[9]),
                ),

                "right_mouth": (
                    float(face[10]),
                    float(face[11]),
                ),

                "left_mouth": (
                    float(face[12]),
                    float(face[13]),
                ),
            }

            results.append(result)

        return results

    def detect_largest(self, frame):
        """
        Return the largest detected face.

        Useful for a driver camera where the driver is
        expected to be the primary visible person.
        """

        faces = self.detect(frame)

        if not faces:
            return None

        return max(
            faces,
            key=lambda face: face["w"] * face["h"],
        )

    @staticmethod
    def crop_face(frame, face, padding=0.20):
        """
        Crop a detected face with configurable padding.
        """

        if frame is None or face is None:
            return None

        height, width = frame.shape[:2]

        x = face["x"]
        y = face["y"]
        w = face["w"]
        h = face["h"]

        pad_x = int(w * padding)
        pad_y = int(h * padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def draw_detection(frame, face):
        """
        Draw the detected face and five YuNet landmarks.
        """

        if frame is None or face is None:
            return frame

        output = frame.copy()

        x = face["x"]
        y = face["y"]
        w = face["w"]
        h = face["h"]

        # Face bounding box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        # Confidence
        label = f"Face {face['confidence']:.2f}"

        cv2.putText(
            output,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        landmarks = [
            ("right_eye", (255, 0, 0)),
            ("left_eye", (0, 0, 255)),
            ("nose", (0, 255, 0)),
            ("right_mouth", (255, 0, 255)),
            ("left_mouth", (0, 255, 255)),
        ]

        for name, color in landmarks:
            point = face[name]

            px = int(round(point[0]))
            py = int(round(point[1]))

            cv2.circle(
                output,
                (px, py),
                4,
                color,
                -1,
            )

        return output


if __name__ == "__main__":
    detector = FaceDetector()

    print("YuNet face detector initialized successfully.")
    print("Model:", detector.model_path)