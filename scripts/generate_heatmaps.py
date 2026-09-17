import os
import re
import cv2
import numpy as np

ROOT = "dataset/Subject01_1_data"

GAZE_DIR = os.path.join(ROOT, "gaze_info")
SCENE_DIR = os.path.join(ROOT, "scene_ims")
OUTPUT_DIR = os.path.join(ROOT, "heatmaps")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_gaze(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Gaze_Loc_2D"):
                nums = re.findall(r"\d+", line)
                if len(nums) >= 2:
                    return int(nums[0]), int(nums[1])
    return None


for gaze_file in sorted(os.listdir(GAZE_DIR)):

    point = read_gaze(os.path.join(GAZE_DIR, gaze_file))

    if point is None:
        continue

    image_name = gaze_file.replace("_gaze.txt", "_scene.png")

    scene_path = os.path.join(SCENE_DIR, image_name)

    scene = cv2.imread(scene_path)

    if scene is None:
        print("Missing:", image_name)
        continue

    h, w = scene.shape[:2]

    heatmap = np.zeros((h, w), dtype=np.float32)

    cv2.circle(
        heatmap,
        point,
        40,
        1,
        -1
    )

    heatmap = cv2.GaussianBlur(
        heatmap,
        (101, 101),
        30
    )

    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap = (heatmap * 255).astype(np.uint8)

    output_name = gaze_file.replace("_gaze.txt", "_heatmap.png")

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, output_name),
        heatmap
    )


print("====================================")
print("Heatmaps Generated Successfully")
print("====================================")