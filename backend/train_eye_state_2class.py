import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = os.path.join(
    os.path.dirname(__file__),
    "eye_dataset"
)

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "checkpoints",
    "eye_state_2class_best.pth"
)

IMAGE_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

VALIDATION_SPLIT = 0.20

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)

print("L-DriX-Net 2-Class Eye-State Retraining")

print("=" * 70)

print()

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        8
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

print(
    f"Dataset: {DATASET_DIR}"
)

full_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

print()

print(
    "Original classes:"
)

for index, name in enumerate(
    full_dataset.classes
):

    print(
        f"  {index} -> {name}"
    )

# ============================================================
# FIND CLOSED / OPEN INDICES
# ============================================================

class_to_index = (
    full_dataset.class_to_idx
)

if "closed" not in class_to_index:

    raise RuntimeError(
        "closed class not found."
    )

if "open" not in class_to_index:

    raise RuntimeError(
        "open class not found."
    )

closed_index = (
    class_to_index["closed"]
)

open_index = (
    class_to_index["open"]
)


# ============================================================
# FILTER DATASET
# ============================================================

selected_indices = []

for index, (_, label) in enumerate(
    full_dataset.samples
):

    if label in (
        closed_index,
        open_index
    ):

        selected_indices.append(
            index
        )


print()

print(
    f"Using {len(selected_indices)} "
    f"closed/open images."
)

print(
    "Ignoring occluded class."
)


# ============================================================
# CREATE CLOSED/OPEN LABELS
# ============================================================

samples = []

for index in selected_indices:

    path, original_label = (
        full_dataset.samples[index]
    )

    if original_label == closed_index:

        new_label = 0

    else:

        new_label = 1

    samples.append(
        (
            path,
            new_label
        )
    )


# ============================================================
# COUNT CLASSES
# ============================================================

closed_count = sum(
    label == 0
    for _, label in samples
)

open_count = sum(
    label == 1
    for _, label in samples
)

print()

print(
    f"CLOSED images: {closed_count}"
)

print(
    f"OPEN images:   {open_count}"
)


# ============================================================
# CUSTOM DATASET
# ============================================================

class EyeDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        samples,
        transform
    ):

        self.samples = samples

        self.transform = transform

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):

        path, label = (
            self.samples[index]
        )

        image = (
            datasets.folder.default_loader(
                path
            )
        )

        image = self.transform(
            image
        )

        return image, label


# ============================================================
# STRATIFIED SPLIT
# ============================================================

closed_indices = [
    i
    for i, (_, label)
    in enumerate(samples)
    if label == 0
]

open_indices = [
    i
    for i, (_, label)
    in enumerate(samples)
    if label == 1
]


random.shuffle(
    closed_indices
)

random.shuffle(
    open_indices
)


closed_val_count = int(
    len(closed_indices)
    *
    VALIDATION_SPLIT
)

open_val_count = int(
    len(open_indices)
    *
    VALIDATION_SPLIT
)


closed_val = (
    closed_indices[
        :closed_val_count
    ]
)

closed_train = (
    closed_indices[
        closed_val_count:
    ]
)

open_val = (
    open_indices[
        :open_val_count
    ]
)

open_train = (
    open_indices[
        open_val_count:
    ]
)


train_indices = (
    closed_train +
    open_train
)

val_indices = (
    closed_val +
    open_val
)


random.shuffle(
    train_indices
)

random.shuffle(
    val_indices
)


train_samples = [
    samples[i]
    for i in train_indices
]

val_samples = [
    samples[i]
    for i in val_indices
]


# ============================================================
# DATASETS
# ============================================================

train_dataset = EyeDataset(
    train_samples,
    train_transform
)

val_dataset = EyeDataset(
    val_samples,
    val_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


print()

print(
    f"Training samples:   {len(train_dataset)}"
)

print(
    f"Validation samples: {len(val_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

print()

print(
    "Loading MobileNetV3-Small..."
)

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

input_features = (
    model.classifier[-1].in_features
)

model.classifier[-1] = nn.Linear(
    input_features,
    2
)

model = model.to(
    DEVICE
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

total = (
    closed_count +
    open_count
)

closed_weight = (
    total /
    (2 * closed_count)
)

open_weight = (
    total /
    (2 * open_count)
)

class_weights = torch.tensor(
    [
        closed_weight,
        open_weight
    ],
    dtype=torch.float32
).to(
    DEVICE
)


print()

print(
    "Class weights:"
)

print(
    f"  CLOSED: {closed_weight:.4f}"
)

print(
    f"  OPEN:   {open_weight:.4f}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

print()

print("=" * 70)

print("STARTING TRAINING")

print("=" * 70)


for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item()
            *
            images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        train_correct += (
            predictions ==
            labels
        ).sum().item()

        train_total += (
            labels.size(0)
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                *
                images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            val_correct += (
                predictions ==
                labels
            ).sum().item()

            val_total += (
                labels.size(0)
            )


    # ========================================================
    # METRICS
    # ========================================================

    train_loss /= train_total

    train_accuracy = (
        train_correct /
        train_total
    )

    val_loss /= val_total

    val_accuracy = (
        val_correct /
        val_total
    )


    scheduler.step()


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


    # ========================================================
    # SAVE BEST
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    [
                        "closed",
                        "open"
                    ],

                "num_classes":
                    2,

                "validation_accuracy":
                    val_accuracy,

                "epoch":
                    epoch + 1
            },
            CHECKPOINT_PATH
        )

        print(
            "  -> Best checkpoint saved."
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()

print("=" * 70)

print("FINAL VALIDATION")

print("=" * 70)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


all_predictions = []

all_labels = []


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(
            DEVICE
        )

        outputs = model(
            images
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.tolist()
        )

        all_labels.extend(
            labels.numpy().tolist()
        )


print()

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "CLOSED",
            "OPEN"
        ],
        digits=4
    )
)


print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)


print()

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print()

print(
    f"Checkpoint saved to:"
)

print(
    CHECKPOINT_PATH
)

print()

print("=" * 70)

print("TRAINING COMPLETE")

print("=" * 70)