import argparse
import csv
import math
from pathlib import Path

from PIL import Image, UnidentifiedImageError


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = MODEL_ROOT / "dataset" / "crowd_counting"
EXPECTED_FRAME_COUNT = 16
EXPECTED_SIZE = (1038, 480)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate crowd point annotations without changing dataset files."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def read_counts(path, errors):
    if not path.is_file():
        errors.append("Missing counts CSV: {}".format(path))
        return {}

    counts = {}
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"frame", "count"}.issubset(reader.fieldnames):
            errors.append("counts.csv must contain frame,count columns.")
            return {}
        for line_number, row in enumerate(reader, start=2):
            frame_name = (row.get("frame") or "").strip()
            if not frame_name:
                errors.append("counts.csv line {} has an empty frame name.".format(line_number))
                continue
            if frame_name in counts:
                errors.append("counts.csv contains duplicate frame: {}".format(frame_name))
                continue
            try:
                count = int(row.get("count", ""))
            except (TypeError, ValueError):
                errors.append("counts.csv line {} has an invalid count.".format(line_number))
                continue
            if count < 0:
                errors.append("counts.csv line {} has a negative count.".format(line_number))
                continue
            counts[frame_name] = count
    return counts


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    images_dir = dataset_root / "images" / "annotated"
    labels_dir = dataset_root / "annotations" / "points"
    errors = []
    counts = read_counts(dataset_root / "annotations" / "counts.csv", errors)
    all_points_path = dataset_root / "annotations" / "all_points.txt"
    combined_counts = {}
    if not all_points_path.is_file():
        errors.append("Missing combined point file: {}".format(all_points_path))
    else:
        with all_points_path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 3:
                    errors.append(
                        "all_points.txt line {} must contain frame_name x y.".format(
                            line_number
                        )
                    )
                    continue
                frame_name = fields[0]
                if frame_name not in counts:
                    errors.append(
                        "all_points.txt line {} references unknown frame {}.".format(
                            line_number, frame_name
                        )
                    )
                    continue
                try:
                    x, y = float(fields[1]), float(fields[2])
                except ValueError:
                    errors.append(
                        "all_points.txt line {} contains a non-numeric coordinate.".format(
                            line_number
                        )
                    )
                    continue
                if not math.isfinite(x) or not math.isfinite(y):
                    errors.append(
                        "all_points.txt line {} contains a non-finite coordinate.".format(
                            line_number
                        )
                    )
                    continue
                combined_counts[frame_name] = combined_counts.get(frame_name, 0) + 1
                if not (0 <= x < EXPECTED_SIZE[0] and 0 <= y < EXPECTED_SIZE[1]):
                    errors.append(
                        "all_points.txt line {} is outside image bounds: {}, {}.".format(
                            line_number, x, y
                        )
                    )

    if len(counts) != EXPECTED_FRAME_COUNT:
        errors.append(
            "Expected {} count rows; found {}.".format(EXPECTED_FRAME_COUNT, len(counts))
        )
    for frame_name, expected_count in sorted(counts.items()):
        combined_count = combined_counts.get(frame_name, 0)
        if combined_count != expected_count:
            errors.append(
                "all_points.txt records {} points for {}; counts.csv records {}.".format(
                    combined_count, frame_name, expected_count
                )
            )

    image_names = {
        path.name for path in images_dir.iterdir()
        if images_dir.is_dir() and path.is_file() and path.suffix.lower() in (".jpg", ".jpeg")
    } if images_dir.is_dir() else set()
    label_names = {
        path.name for path in labels_dir.iterdir()
        if labels_dir.is_dir() and path.is_file() and path.suffix.lower() == ".txt"
    } if labels_dir.is_dir() else set()

    if not images_dir.is_dir():
        errors.append("Missing image directory: {}".format(images_dir))
    if not labels_dir.is_dir():
        errors.append("Missing point-label directory: {}".format(labels_dir))
    if len(image_names) != EXPECTED_FRAME_COUNT:
        errors.append(
            "Expected {} images; found {}.".format(EXPECTED_FRAME_COUNT, len(image_names))
        )
    if len(label_names) != EXPECTED_FRAME_COUNT:
        errors.append(
            "Expected {} point files; found {}.".format(EXPECTED_FRAME_COUNT, len(label_names))
        )

    expected_image_names = set(counts)
    expected_label_names = {Path(name).stem + ".txt" for name in counts}
    for name in sorted(expected_image_names - image_names):
        errors.append("Missing image: {}".format(name))
    for name in sorted(expected_label_names - label_names):
        errors.append("Missing label: {}".format(name))
    for name in sorted(image_names - expected_image_names):
        errors.append("Unexpected image: {}".format(name))
    for name in sorted(label_names - expected_label_names):
        errors.append("Unexpected label: {}".format(name))

    print("Dataset: {}".format(dataset_root))
    print("Images: {} / {}".format(len(image_names), EXPECTED_FRAME_COUNT))
    print("Point files: {} / {}".format(len(label_names), EXPECTED_FRAME_COUNT))
    print("Combined points: {}".format(sum(combined_counts.values())))

    for frame_name, expected_count in sorted(counts.items()):
        image_path = images_dir / frame_name
        label_path = labels_dir / (Path(frame_name).stem + ".txt")
        if image_path.is_file():
            try:
                with Image.open(str(image_path)) as image:
                    if image.size != EXPECTED_SIZE:
                        errors.append(
                            "{} has dimensions {}; expected {}.".format(
                                frame_name, image.size, EXPECTED_SIZE
                            )
                        )
            except (OSError, UnidentifiedImageError) as error:
                errors.append("Cannot read image {}: {}".format(frame_name, error))

        if not label_path.is_file():
            continue

        point_count = 0
        with label_path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 2:
                    errors.append(
                        "{} line {} must contain exactly x y.".format(
                            label_path.name, line_number
                        )
                    )
                    continue
                try:
                    x, y = float(fields[0]), float(fields[1])
                except ValueError:
                    errors.append(
                        "{} line {} contains a non-numeric coordinate.".format(
                            label_path.name, line_number
                        )
                    )
                    continue
                if not math.isfinite(x) or not math.isfinite(y):
                    errors.append(
                        "{} line {} contains a non-finite coordinate.".format(
                            label_path.name, line_number
                        )
                    )
                    continue
                point_count += 1
                if not (0 <= x < EXPECTED_SIZE[0] and 0 <= y < EXPECTED_SIZE[1]):
                    errors.append(
                        "{} line {} is outside image bounds: {}, {}.".format(
                            label_path.name, line_number, x, y
                        )
                    )

        print(
            "{}: {} points (counts.csv: {})".format(
                frame_name, point_count, expected_count
            )
        )
        if point_count != expected_count:
            errors.append(
                "{} has {} points but counts.csv records {}.".format(
                    frame_name, point_count, expected_count
                )
            )

    if errors:
        print("Validation failed with {} issue(s):".format(len(errors)))
        for error in errors:
            print("ERROR: {}".format(error))
        return 1

    print("Validation passed: all {} frames, dimensions, coordinates, and counts agree.".format(
        EXPECTED_FRAME_COUNT
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
