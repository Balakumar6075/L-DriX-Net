import os
import random
import shutil

random.seed(42)

SOURCE = "dataset/Subject01_1_data"

TRAIN = "dataset/train"
VAL = "dataset/val"
TEST = "dataset/test"

folders = [
    "face_ims",
    "scene_ims",
    "gaze_info"
]

# Create destination folders
for split in [TRAIN, VAL, TEST]:
    for folder in folders:
        os.makedirs(os.path.join(split, folder), exist_ok=True)

# Read image list
images = sorted(os.listdir(os.path.join(SOURCE, "face_ims")))

# Shuffle images
random.shuffle(images)

total = len(images)

train_size = int(0.7 * total)
val_size = int(0.15 * total)

train_images = images[:train_size]
val_images = images[train_size:train_size + val_size]
test_images = images[train_size + val_size:]


def copy_files(image_list, destination):

    for image in image_list:

        base = os.path.splitext(image)[0]

        # Face image
        face_src = os.path.join(SOURCE, "face_ims", image)
        face_dst = os.path.join(destination, "face_ims", image)

        # Scene image
        scene_src = os.path.join(SOURCE, "scene_ims", image)
        scene_dst = os.path.join(destination, "scene_ims", image)

        # Gaze file
        gaze_src = os.path.join(
            SOURCE,
            "gaze_info",
            base + "_gaze.txt"
        )

        gaze_dst = os.path.join(
            destination,
            "gaze_info",
            base + "_gaze.txt"
        )

        if os.path.exists(face_src):
            shutil.copy2(face_src, face_dst)

        if os.path.exists(scene_src):
            shutil.copy2(scene_src, scene_dst)

        if os.path.exists(gaze_src):
            shutil.copy2(gaze_src, gaze_dst)


copy_files(train_images, TRAIN)
copy_files(val_images, VAL)
copy_files(test_images, TEST)

print("=" * 40)
print("Dataset Split Completed")
print("=" * 40)
print(f"Total Images : {total}")
print(f"Train Images : {len(train_images)}")
print(f"Val Images   : {len(val_images)}")
print(f"Test Images  : {len(test_images)}")
print("=" * 40)