import argparse
import csv
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = MODEL_ROOT / "dataset" / "crowd_counting"
TRAIN_LIST_NAME = "shanghai_tech_part_a_train.list"
VALIDATION_LIST_NAME = "shanghai_tech_part_a_test.list"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create P2PNet list files while leaving image and point files unchanged."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--validation-every",
        type=int,
        default=4,
        help="Place every Nth sorted frame in the validation list (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated list files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validation_every < 2:
        raise ValueError("--validation-every must be at least 2.")

    dataset_root = args.dataset_root
    counts_path = dataset_root / "annotations" / "counts.csv"
    if not counts_path.is_file():
        raise FileNotFoundError("Counts CSV does not exist: {}".format(counts_path))

    with counts_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"frame", "count"}.issubset(reader.fieldnames):
            raise ValueError("counts.csv must contain frame,count columns.")
        frame_names = sorted(
            (row.get("frame") or "").strip() for row in reader
        )
    if len(frame_names) != 16 or any(not name for name in frame_names):
        raise ValueError("Expected exactly 16 non-empty frame entries in counts.csv.")
    if len(set(frame_names)) != len(frame_names):
        raise ValueError("counts.csv contains duplicate frame names.")

    image_dir = dataset_root / "images" / "annotated"
    points_dir = dataset_root / "annotations" / "points"
    training = []
    validation = []
    for index, frame_name in enumerate(frame_names, start=1):
        image_path = image_dir / frame_name
        point_path = points_dir / (Path(frame_name).stem + ".txt")
        if not image_path.is_file():
            raise FileNotFoundError("Image does not exist: {}".format(image_path))
        if not point_path.is_file():
            raise FileNotFoundError("Point annotation does not exist: {}".format(point_path))
        entry = "{}/{} {}/{}".format(
            "images/annotated",
            frame_name,
            "annotations/points",
            point_path.name,
        )
        if index % args.validation_every == 0:
            validation.append(entry)
        else:
            training.append(entry)

    output_files = {
        dataset_root / TRAIN_LIST_NAME: training,
        dataset_root / VALIDATION_LIST_NAME: validation,
    }
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Generated list file(s) already exist; pass --overwrite to replace: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    for path, entries in output_files.items():
        path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        print("Wrote {} entries to {}".format(len(entries), path))
    print(
        "P2PNet list split created. The frames are a single preliminary sequence; "
        "this holdout is not an independent validation set."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
