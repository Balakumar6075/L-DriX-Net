from utils.temporal_dataset import TemporalGazeDataset


SUBJECT_PATH = (
    "dataset/Subject01_1_data"
)


def main():

    print("=" * 60)
    print("Temporal Dataset Test")
    print("=" * 60)

    dataset = TemporalGazeDataset(
        subject_path=SUBJECT_PATH,
        temporal_length=16
    )

    print()
    print("Total synchronized frames:")
    print(len(dataset.frame_ids))

    print()
    print("Temporal length:")
    print(dataset.temporal_length)

    print()
    print("Number of temporal windows:")
    print(len(dataset))

    # --------------------------------------------------------
    # Load first temporal sample
    # --------------------------------------------------------

    sample = dataset[0]

    print()
    print("FIRST SAMPLE")
    print("-" * 60)

    print(
        "Face:",
        sample["face"].shape
    )

    print(
        "Scene:",
        sample["scene"].shape
    )

    print(
        "Gaze:",
        sample["gaze"].shape
    )

    print(
        "Heatmap:",
        sample["heatmap"].shape
    )

    print()
    print(
        "Frame IDs:",
        sample["frame_ids"]
    )

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()