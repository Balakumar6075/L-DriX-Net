import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TemporalGazeDataset(Dataset):
    """
    Temporal dataset for L-DriX-Net.

    Each sample contains a sequence of consecutive
    synchronized dataset entries.

    Input:
        16 face images
        16 scene images
        16 gaze vectors
        16 heatmaps

    Output:
        face_sequence   -> [T, 3, 224, 224]
        scene_sequence  -> [T, 3, 224, 224]
        gaze_sequence   -> [T, 24]
        heatmap_sequence-> [T, 1, 224, 224]
        frame_ids       -> list of frame IDs
    """

    def __init__(
        self,
        subject_path,
        temporal_length=16,
        transform=None
    ):

        self.subject_path = subject_path
        self.temporal_length = temporal_length

        self.face_dir = os.path.join(
            subject_path,
            "face_ims"
        )

        self.scene_dir = os.path.join(
            subject_path,
            "scene_ims"
        )

        self.gaze_dir = os.path.join(
            subject_path,
            "gaze_info"
        )

        self.heatmap_dir = os.path.join(
            subject_path,
            "heatmaps"
        )

        # ----------------------------------------------------
        # Image transformation
        # ----------------------------------------------------

        if transform is None:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor()
            ])

        else:

            self.transform = transform

        # ----------------------------------------------------
        # Discover synchronized frame IDs
        # ----------------------------------------------------

        self.frame_ids = self._find_synchronized_frames()

        # ----------------------------------------------------
        # Create temporal windows
        # ----------------------------------------------------

        self.windows = []

        for i in range(
            len(self.frame_ids)
            - temporal_length
            + 1
        ):

            window = self.frame_ids[
                i:i + temporal_length
            ]

            self.windows.append(
                window
            )

    # ========================================================
    # Find synchronized frames
    # ========================================================

    def _find_synchronized_frames(self):

        face_ids = self._get_ids(
            self.face_dir,
            "_face.png"
        )

        scene_ids = self._get_ids(
            self.scene_dir,
            "_scene.png"
        )

        gaze_ids = self._get_ids(
            self.gaze_dir,
            "_gaze.txt"
        )

        heatmap_ids = self._get_ids(
            self.heatmap_dir,
            "_heatmap.png"
        )

        # Only use frames present in ALL four sources.

        synchronized = sorted(
            face_ids
            & scene_ids
            & gaze_ids
            & heatmap_ids
        )

        return synchronized

    # ========================================================
    # Extract numeric frame IDs
    # ========================================================

    @staticmethod
    def _get_ids(directory, suffix):

        ids = set()

        if not os.path.isdir(directory):
            return ids

        for filename in os.listdir(directory):

            if not filename.endswith(suffix):
                continue

            frame_name = filename[
                :-len(suffix)
            ]

            if not re.fullmatch(
                r"\d+",
                frame_name
            ):
                continue

            ids.add(
                int(frame_name)
            )

        return ids

    # ========================================================
    # Parse gaze annotation
    # ========================================================

    @staticmethod
    def _parse_vector(
        text,
        key,
        expected_values
    ):

        pattern = rf"{key}:\s*\[([^\]]+)\]"

        match = re.search(
            pattern,
            text
        )

        if match is None:

            return []

        values = match.group(1)

        values = re.split(
            r"[\s,]+",
            values.strip()
        )

        values = [
            float(v)
            for v in values
            if v
        ]

        if len(values) != expected_values:

            return []

        return values

    def _load_gaze(self, frame_id):

        filename = os.path.join(
            self.gaze_dir,
            f"{frame_id:08d}_gaze.txt"
        )

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        values = []

        # ----------------------------------------------------
        # Gaze location
        # ----------------------------------------------------

        values.extend(
            self._parse_vector(
                text,
                "Gaze_Loc_2D",
                2
            )
        )

        values.extend(
            self._parse_vector(
                text,
                "Gaze_Loc_3D",
                3
            )
        )

        # ----------------------------------------------------
        # Left gaze
        # ----------------------------------------------------

        values.extend(
            self._parse_vector(
                text,
                "Left_Gaze_Dir",
                3
            )
        )

        # ----------------------------------------------------
        # Right gaze
        # ----------------------------------------------------

        values.extend(
            self._parse_vector(
                text,
                "Right_Gaze_Dir",
                3
            )
        )

        # ----------------------------------------------------
        # Existing model expects 24 dimensions
        # ----------------------------------------------------

        if len(values) < 24:

            values.extend(
                [0.0] * (
                    24 - len(values)
                )
            )

        elif len(values) > 24:

            values = values[:24]

        return torch.tensor(
            values,
            dtype=torch.float32
        )

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(self):

        return len(self.windows)

    # ========================================================
    # Load one temporal sample
    # ========================================================

    def __getitem__(self, index):

        frame_ids = self.windows[index]

        face_frames = []

        scene_frames = []

        gaze_frames = []

        heatmap_frames = []

        # ----------------------------------------------------
        # Load all T frames
        # ----------------------------------------------------

        for frame_id in frame_ids:

            face_path = os.path.join(
                self.face_dir,
                f"{frame_id:08d}_face.png"
            )

            scene_path = os.path.join(
                self.scene_dir,
                f"{frame_id:08d}_scene.png"
            )

            heatmap_path = os.path.join(
                self.heatmap_dir,
                f"{frame_id:08d}_heatmap.png"
            )

            # ----------------------------------------------
            # Face
            # ----------------------------------------------

            face_image = Image.open(
                face_path
            ).convert("RGB")

            face_image = self.transform(
                face_image
            )

            face_frames.append(
                face_image
            )

            # ----------------------------------------------
            # Scene
            # ----------------------------------------------

            scene_image = Image.open(
                scene_path
            ).convert("RGB")

            scene_image = self.transform(
                scene_image
            )

            scene_frames.append(
                scene_image
            )

            # ----------------------------------------------
            # Gaze
            # ----------------------------------------------

            gaze = self._load_gaze(
                frame_id
            )

            gaze_frames.append(
                gaze
            )

            # ----------------------------------------------
            # Heatmap
            # ----------------------------------------------

            heatmap = Image.open(
                heatmap_path
            ).convert("L")

            heatmap = self.transform(
                heatmap
            )

            heatmap_frames.append(
                heatmap
            )

        # ----------------------------------------------------
        # Stack temporal dimension
        # ----------------------------------------------------

        face_sequence = torch.stack(
            face_frames,
            dim=0
        )

        scene_sequence = torch.stack(
            scene_frames,
            dim=0
        )

        gaze_sequence = torch.stack(
            gaze_frames,
            dim=0
        )

        heatmap_sequence = torch.stack(
            heatmap_frames,
            dim=0
        )

        return {
            "face": face_sequence,
            "scene": scene_sequence,
            "gaze": gaze_sequence,
            "heatmap": heatmap_sequence,
            "frame_ids": frame_ids
        }