import os
import re

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LBWDataset(Dataset):

    def __init__(self, root_dir):

        self.face_dir = os.path.join(root_dir, "face_ims")
        self.scene_dir = os.path.join(root_dir, "scene_ims")
        self.gaze_dir = os.path.join(root_dir, "gaze_info")
        self.heatmap_dir = os.path.join(root_dir, "heatmaps")

        self.face_images = sorted(os.listdir(self.face_dir))
        self.scene_images = sorted(os.listdir(self.scene_dir))
        self.gaze_files = sorted(os.listdir(self.gaze_dir))
        self.heatmap_images = sorted(os.listdir(self.heatmap_dir))

        assert len(self.face_images) == len(self.scene_images)
        assert len(self.face_images) == len(self.gaze_files)
        assert len(self.face_images) == len(self.heatmap_images)

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.heatmap_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.face_images)

    def parse_vector(self, text):

        nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', text)

        return [float(x) for x in nums]

    def read_gaze(self, file_path):

        gaze = {}

        with open(file_path, "r") as f:

            for line in f:

                if ":" not in line:
                    continue

                key, value = line.split(":", 1)

                gaze[key.strip()] = self.parse_vector(value)

        return gaze

    def gaze_to_tensor(self, gaze):

        keys = [
            "Gaze_Loc_2D",
            "Gaze_Loc_3D",
            "Left_Gaze_Dir",
            "Right_Gaze_Dir",
            "Left_2D_Eye_Loc",
            "Right_2D_Eye_Loc",
            "Left_3D_Eye_Loc",
            "Right_3D_Eye_Loc"
        ]

        vector = []

        for key in keys:

            if key in gaze:
                vector.extend(gaze[key])

        # Ensure fixed length = 24
        if len(vector) < 24:
            vector.extend([0.0] * (24 - len(vector)))

        elif len(vector) > 24:
            vector = vector[:24]

        return torch.tensor(vector, dtype=torch.float32)

    def __getitem__(self, idx):

        face_path = os.path.join(
            self.face_dir,
            self.face_images[idx]
        )

        scene_path = os.path.join(
            self.scene_dir,
            self.scene_images[idx]
        )

        gaze_path = os.path.join(
            self.gaze_dir,
            self.gaze_files[idx]
        )

        heatmap_path = os.path.join(
            self.heatmap_dir,
            self.heatmap_images[idx]
        )

        face = Image.open(face_path).convert("RGB")
        scene = Image.open(scene_path).convert("RGB")
        heatmap = Image.open(heatmap_path).convert("L")

        face = self.image_transform(face)
        scene = self.image_transform(scene)
        heatmap = self.heatmap_transform(heatmap)

        gaze = self.read_gaze(gaze_path)
        gaze = self.gaze_to_tensor(gaze)

        return face, scene, gaze, heatmap