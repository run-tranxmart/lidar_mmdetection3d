import argparse
from pathlib import Path
from random import shuffle


def split_set_generator(
    bin_path: Path, split_set_dir: Path, train_ratio: int = 90, val_ratio=10
):
    """
    Generate a train-val split for a set of images in a directory
    :param bin_path: Path to the directory containing the images
    :param split_set_dir: Path to the directory where the train and val sets will be saved
    :param train_ratio: Percentage of images to be used for training
    :param val_ratio: Percentage of images to be used for validation
    :return: None
    """
    assert train_ratio + val_ratio == 100, "Train and validation ratios must sum to 100"
    assert bin_path.is_dir(), "Invalid directory path"
    split_set_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of images
    cloud_bin = Path(bin_path).glob("*.bin")
    cloud_bin = list(cloud_bin)
    cloud_bin.sort()

    # Shuffle the list of cloud bins
    shuffle(cloud_bin)
    # Just put the name of the file in the list not complete path
    cloud_bin = [Path(file).name for file in cloud_bin]

    # Split the list of cloud bins
    train_split = int(len(cloud_bin) * train_ratio / 100)
    val_split = int(len(cloud_bin) * val_ratio / 100)

    # Generate the train and val sets
    train_set = cloud_bin[:train_split]
    val_set = cloud_bin[train_split : train_split + val_split]

    # Write the train and val sets to files
    with open(split_set_dir / "train.txt", "w") as f:
        for item in train_set:
            f.write("%s\n" % item)

    with open(split_set_dir / "val.txt", "w") as f:
        for item in val_set:
            f.write("%s\n" % item)

    with open(split_set_dir / "trainval.txt", "w") as f:
        for item in cloud_bin:
            f.write("%s\n" % item)

    with open(split_set_dir / "test.txt", "w") as f:
        for item in val_set[:1000]:
            f.write("%s\n" % item)

    print(f"Train set: {len(train_set)} images")
    print(f"Validation set: {len(val_set)} images")
    print(f"Total: {len(cloud_bin)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bin_path",
        type=str,
        required=True,
        help="Path to the directory containing the images",
    )

    parser.add_argument(
        "--split_set_dir",
        type=str,
        required=True,
        help="Path to the directory where the train and val sets will be saved",
    )

    parser.add_argument(
        "--train_ratio",
        type=int,
        default=90,
        help="Percentage of images to be used for training",
    )

    parser.add_argument(
        "--val_ratio",
        type=int,
        default=10,
        help="Percentage of images to be used for validation",
    )
    args = parser.parse_args()

    split_set_generator(
        Path(args.bin_path), Path(args.split_set_dir), args.train_ratio, args.val_ratio
    )
