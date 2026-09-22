import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from detection.eye_state_model import EyeStateCNN


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "eye_state_dataset"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "eye_state_best.pth"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 15

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
        degrees=8
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
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

def load_dataset():

    dataset = datasets.ImageFolder(
        DATASET_DIR,
        transform=train_transform
    )

    print()
    print("Classes detected by ImageFolder:")

    for index, name in enumerate(
        dataset.classes
    ):

        print(
            f"{index} -> {name}"
        )

    return dataset


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    dataset
):

    targets = np.array(
        dataset.targets
    )

    class_counts = np.bincount(
        targets
    )

    print()
    print("Class distribution:")

    for index, count in enumerate(
        class_counts
    ):

        print(
            f"Class {index}: {count}"
        )

    total = len(targets)

    num_classes = len(
        class_counts
    )

    weights = (
        total
        /
        (
            num_classes
            * class_counts
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(
    model,
    loader,
    num_classes
):

    matrix = np.zeros(
        (
            num_classes,
            num_classes
        ),
        dtype=np.int64
    )

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            for actual, predicted in zip(
                labels.numpy(),
                predictions.cpu().numpy()
            ):

                matrix[
                    actual,
                    predicted
                ] += 1

    return matrix


# ============================================================
# METRICS
# ============================================================

def print_metrics(
    matrix,
    class_names
):

    print()
    print("=" * 65)
    print("VALIDATION METRICS")
    print("=" * 65)

    total_correct = (
        np.trace(matrix)
    )

    total_samples = (
        matrix.sum()
    )

    accuracy = (
        total_correct
        /
        max(total_samples, 1)
    )

    print(
        f"Overall Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()

    for i, class_name in enumerate(
        class_names
    ):

        true_positive = (
            matrix[i, i]
        )

        false_positive = (
            matrix[:, i].sum()
            - true_positive
        )

        false_negative = (
            matrix[i, :].sum()
            - true_positive
        )

        precision = (
            true_positive
            /
            max(
                true_positive
                + false_positive,
                1
            )
        )

        recall = (
            true_positive
            /
            max(
                true_positive
                + false_negative,
                1
            )
        )

        f1 = (
            2
            * precision
            * recall
            /
            max(
                precision + recall,
                1e-8
            )
        )

        print(
            f"{class_name.upper():10s} "
            f"Precision: {precision:.4f}  "
            f"Recall: {recall:.4f}  "
            f"F1: {f1:.4f}"
        )

    print()
    print("Confusion Matrix:")
    print()

    header = (
        "Actual \\ Pred"
        + "".join(
            f"{name:>12}"
            for name in class_names
        )
    )

    print(header)

    for i, name in enumerate(
        class_names
    ):

        row = (
            f"{name:12s}"
            + "".join(
                f"{matrix[i, j]:12d}"
                for j in range(
                    len(class_names)
                )
            )
        )

        print(row)

    print("=" * 65)


# ============================================================
# TRAIN
# ============================================================

def main():

    print("=" * 65)
    print("L-DriX-Net 3-Class Eye-State Training")
    print("=" * 65)

    print()
    print(
        f"Dataset: {DATASET_DIR}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    full_dataset = load_dataset()

    class_names = (
        full_dataset.classes
    )

    num_classes = len(
        class_names
    )

    # --------------------------------------------------------
    # VERIFY CLASSES
    # --------------------------------------------------------

    expected_classes = {
        "closed",
        "open",
        "occluded"
    }

    actual_classes = set(
        class_names
    )

    if actual_classes != expected_classes:

        raise RuntimeError(
            "\nExpected dataset classes:\n"
            "  closed\n"
            "  open\n"
            "  occluded\n"
            "\nFound:\n"
            f"  {class_names}"
        )

    if num_classes != 3:

        raise RuntimeError(
            "Expected exactly 3 classes."
        )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            full_dataset
        )
    )

    class_weights = (
        class_weights.to(
            DEVICE
        )
    )

    print()
    print(
        "Class weights:"
    )

    print(
        class_weights
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    validation_size = int(
        len(full_dataset)
        * VALIDATION_SPLIT
    )

    training_size = (
        len(full_dataset)
        - validation_size
    )

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_dataset, val_dataset = (
        random_split(
            full_dataset,
            [
                training_size,
                validation_size
            ],
            generator=generator
        )
    )

    # --------------------------------------------------------
    # VALIDATION TRANSFORM
    # --------------------------------------------------------

    val_dataset.dataset.transform = (
        val_transform
    )

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

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
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = EyeStateCNN(
        pretrained=True
    )

    model = model.to(
        DEVICE
    )

    print()
    print(
        "Model:"
    )

    print(
        "MobileNetV3-Small"
    )

    print(
        f"Output classes: "
        f"{num_classes}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_accuracy = 0.0

    print()
    print("=" * 65)
    print("Starting training...")
    print("=" * 65)

    for epoch in range(
        EPOCHS
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in (
            train_loader
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
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

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            correct += (
                (
                    predictions
                    == labels
                )
                .sum()
                .item()
            )

            total += (
                labels.size(0)
            )

        train_loss = (
            running_loss
            /
            max(total, 1)
        )

        train_accuracy = (
            correct
            /
            max(total, 1)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        validation_loss = 0.0

        val_correct = 0

        val_total = 0

        with torch.no_grad():

            for images, labels in (
                val_loader
            ):

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True
                )

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

                validation_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = (
                    torch.argmax(
                        outputs,
                        dim=1
                    )
                )

                val_correct += (
                    (
                        predictions
                        == labels
                    )
                    .sum()
                    .item()
                )

                val_total += (
                    labels.size(0)
                )

        val_loss = (
            validation_loss
            /
            max(val_total, 1)
        )

        val_accuracy = (
            val_correct
            /
            max(val_total, 1)
        )

        scheduler.step()

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: "
            f"{val_accuracy * 100:.2f}%"
        )

        # ====================================================
        # SAVE BEST
        # ====================================================

        if val_accuracy > best_accuracy:

            best_accuracy = (
                val_accuracy
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        class_names,

                    "num_classes":
                        num_classes,

                    "validation_accuracy":
                        val_accuracy,

                    "epoch":
                        epoch + 1
                },
                BEST_MODEL_PATH
            )

            print(
                f"  -> New best model saved "
                f"({val_accuracy * 100:.2f}%)"
            )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print()
    print(
        "Loading best checkpoint..."
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    matrix = (
        calculate_confusion_matrix(
            model,
            val_loader,
            num_classes
        )
    )

    print_metrics(
        matrix,
        class_names
    )

    print()
    print("=" * 65)

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Checkpoint: "
        f"{BEST_MODEL_PATH}"
    )

    print("=" * 65)


if __name__ == "__main__":

    main()