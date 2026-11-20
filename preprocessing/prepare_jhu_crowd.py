import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = MODEL_ROOT / "dataset" / "JHU-CROWD++"
SPLITS = {
    "train": 2272,
    "val": 500,
    "test": 1600,
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate JHU-CROWD++ point labels and prepare count/list manifests."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def read_image_labels(path, errors):
    labels = {}
    if not path.is_file():
        errors.append("Missing image-level labels: {}".format(path))
        return labels

    with path.open("r", newline="", encoding="utf-8-sig") as source:
        for line_number, row in enumerate(csv.reader(source), 1):
            if not row:
                continue
            if len(row) != 5:
                errors.append(
                    "{} line {} must contain five comma-separated fields.".format(
                        path, line_number
                    )
                )
                continue
            image_id = row[0].strip()
            try:
                count = int(row[1].strip())
                weather = int(row[3].strip())
                distractor = int(row[4].strip())
            except ValueError:
                errors.append(
                    "{} line {} has invalid numeric metadata.".format(
                        path, line_number
                    )
                )
                continue
            if not image_id:
                errors.append(
                    "{} line {} has an empty image name.".format(path, line_number)
                )
                continue
            if image_id in labels:
                errors.append(
                    "{} contains duplicate image id {}.".format(path, image_id)
                )
                continue
            if count < 0:
                errors.append(
                    "{} line {} has a negative count.".format(path, line_number)
                )
                continue
            if weather not in {0, 1, 2, 3}:
                errors.append(
                    "{} line {} has an invalid weather condition.".format(
                        path, line_number
                    )
                )
            if distractor not in {0, 1}:
                errors.append(
                    "{} line {} has an invalid distractor flag.".format(
                        path, line_number
                    )
                )
            labels[image_id] = {
                "count": count,
                "scene_type": row[2].strip(),
                "weather_condition": weather,
                "distractor": distractor,
            }
    return labels


def read_point_annotations(
    path,
    image_width,
    image_height,
    errors,
    coordinate_errors,
    coordinate_errors_by_split,
    metadata_warnings,
):
    point_count = 0
    if not path.is_file():
        errors.append("Missing point annotation: {}".format(path))
        return point_count

    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                errors.append(
                    "{} line {} must contain x y w h occlusion blur.".format(
                        path, line_number
                    )
                )
                continue
            try:
                x, y, width, height = (float(value) for value in fields[:4])
                occlusion, blur = (int(value) for value in fields[4:])
            except ValueError:
                errors.append(
                    "{} line {} contains a non-numeric annotation.".format(
                        path, line_number
                    )
                )
                continue
            point_count += 1
            if not math.isfinite(x) or not math.isfinite(y):
                errors.append(
                    "{} line {} contains a non-finite point coordinate.".format(
                        path, line_number
                    )
                )
                continue
            if not (0 <= x < image_width and 0 <= y < image_height):
                coordinate_errors.append(
                    "{} line {} point is outside image bounds: {}, {}.".format(
                        path, line_number, x, y
                    )
                )
                coordinate_errors_by_split[path.parent.parent.name] += 1
            if not math.isfinite(width) or not math.isfinite(height):
                metadata_warnings["non_finite_box_metadata_rows"] += 1
            elif width <= 0 or height <= 0:
                metadata_warnings["non_positive_box_metadata_rows"] += 1
                if width < 0 or height < 0:
                    metadata_warnings["negative_box_metadata_rows"] += 1
            if occlusion not in {1, 2, 3}:
                metadata_warnings["invalid_occlusion_rows"] += 1
            if blur not in {0, 1}:
                metadata_warnings["invalid_blur_rows"] += 1
    return point_count


def validate_split(
    dataset_root,
    split,
    expected_count,
    errors,
    coordinate_errors,
    coordinate_errors_by_split,
    metadata_warnings,
):
    split_root = dataset_root / split
    image_dir = split_root / "images"
    annotation_dir = split_root / "gt"
    if not image_dir.is_dir():
        errors.append("Missing image directory: {}".format(image_dir))
        image_paths = []
    else:
        image_paths = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not annotation_dir.is_dir():
        errors.append("Missing annotation directory: {}".format(annotation_dir))
        annotation_paths = []
    else:
        annotation_paths = sorted(annotation_dir.glob("*.txt"))

    image_by_id = {}
    for image_path in image_paths:
        if image_path.stem in image_by_id:
            errors.append(
                "{} contains duplicate image id {}.".format(split, image_path.stem)
            )
        image_by_id[image_path.stem] = image_path

    annotation_by_id = {path.stem: path for path in annotation_paths}
    labels = read_image_labels(split_root / "image_labels.txt", errors)
    if len(image_paths) != expected_count:
        errors.append(
            "{} has {} images; the dataset README specifies {}.".format(
                split, len(image_paths), expected_count
            )
        )

    for image_id in sorted(set(labels) - set(image_by_id)):
        errors.append("{} image label has no image: {}.".format(split, image_id))
    for image_id in sorted(set(image_by_id) - set(labels)):
        errors.append("{} image has no image-level label: {}.".format(split, image_id))
    for image_id in sorted(set(labels) - set(annotation_by_id)):
        errors.append("{} image label has no point file: {}.".format(split, image_id))
    for image_id in sorted(set(annotation_by_id) - set(labels)):
        errors.append("{} point file has no image-level label: {}.".format(split, image_id))

    counts = []
    point_total = 0
    for image_id in sorted(labels):
        image_path = image_by_id.get(image_id)
        annotation_path = annotation_by_id.get(image_id)
        if image_path is None or annotation_path is None:
            continue
        try:
            with Image.open(str(image_path)) as image:
                image.verify()
            with Image.open(str(image_path)) as image:
                width, height = image.size
        except (OSError, ValueError) as error:
            errors.append("Invalid image {}: {}".format(image_path, error))
            continue

        point_count = read_point_annotations(
            annotation_path,
            width,
            height,
            errors,
            coordinate_errors,
            coordinate_errors_by_split,
            metadata_warnings,
        )
        point_total += point_count
        image_count = labels[image_id]["count"]
        if point_count != image_count:
            errors.append(
                "{} has {} point rows but image_labels.txt records {}.".format(
                    annotation_path, point_count, image_count
                )
            )
        counts.append(
            {
                "frame": "{}/images/{}".format(split, image_path.name),
                "count": point_count,
                "split": split,
                "scene_type": labels[image_id]["scene_type"],
                "weather_condition": labels[image_id]["weather_condition"],
                "distractor": labels[image_id]["distractor"],
                "image_label_count": image_count,
            }
        )

    return {
        "image_count": len(image_paths),
        "annotation_file_count": len(annotation_paths),
        "image_label_count": len(labels),
        "point_count": point_total,
        "counts": counts,
        "train_list": [
            (
                "{}/images/{}".format(split, image_by_id[image_id].name),
                "{}/gt/{}.txt".format(split, image_id),
            )
            for image_id in sorted(labels)
            if image_id in image_by_id and image_id in annotation_by_id
        ],
    }


def write_outputs(dataset_root, split_results, report):
    annotations_dir = dataset_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    counts_path = annotations_dir / "counts.csv"
    count_fields = [
        "frame",
        "count",
        "split",
        "scene_type",
        "weather_condition",
        "distractor",
        "image_label_count",
    ]
    with counts_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=count_fields)
        writer.writeheader()
        for split in SPLITS:
            writer.writerows(split_results[split]["counts"])

    list_files = {
        "shanghai_tech_part_a_train.list": "train",
        "shanghai_tech_part_a_test.list": "val",
        "jhu_crowd_test.list": "test",
    }
    for filename, split in list_files.items():
        list_path = dataset_root / filename
        with list_path.open("w", encoding="utf-8", newline="\n") as destination:
            for image_path, annotation_path in split_results[split]["train_list"]:
                destination.write("{} {}\n".format(image_path, annotation_path))

    report["prepared_files"] = [
        str(counts_path),
        *[str(dataset_root / filename) for filename in list_files],
    ]


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: {}".format(dataset_root))

    errors = []
    coordinate_errors = []
    coordinate_errors_by_split = {split: 0 for split in SPLITS}
    metadata_warnings = {
        "non_positive_box_metadata_rows": 0,
        "negative_box_metadata_rows": 0,
        "non_finite_box_metadata_rows": 0,
        "invalid_occlusion_rows": 0,
        "invalid_blur_rows": 0,
    }
    split_results = {}
    for split, expected_count in SPLITS.items():
        split_results[split] = validate_split(
            dataset_root,
            split,
            expected_count,
            errors,
            coordinate_errors,
            coordinate_errors_by_split,
            metadata_warnings,
        )

    status = (
        "invalid"
        if errors or coordinate_errors
        else "valid_with_warnings"
        if any(metadata_warnings.values())
        else "valid"
    )
    report = {
        "dataset": "JHU-CROWD++ v2.0",
        "dataset_root": str(dataset_root),
        "status": status,
        "expected_splits": SPLITS,
        "splits": {
            split: {
                key: value
                for key, value in result.items()
                if key != "counts" and key != "train_list"
            }
            for split, result in split_results.items()
        },
        "total_images": sum(
            result["image_count"] for result in split_results.values()
        ),
        "total_points": sum(
            result["point_count"] for result in split_results.values()
        ),
        "error_count": len(errors),
        "errors": errors[:200],
        "point_coordinate_error_count": len(coordinate_errors),
        "point_coordinate_errors_by_split": coordinate_errors_by_split,
        "point_coordinate_errors": coordinate_errors[:200],
        "annotation_metadata_warnings": metadata_warnings,
        "license_note": "Academic and non-commercial use only; see the included License.",
    }
    report_path = dataset_root / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")

    if errors:
        raise ValueError(
            "Dataset structure or count validation found {} issues; see {}.".format(
                len(errors), report_path
            )
        )

    write_outputs(dataset_root, split_results, report)
    with report_path.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")

    for split in SPLITS:
        result = split_results[split]
        print(
            "{}: {} images, {} point annotations".format(
                split, result["image_count"], result["point_count"]
            )
        )
    print("Validated {} images and {} points.".format(
        report["total_images"], report["total_points"]
    ))
    if coordinate_errors:
        print(
            "Found {} point coordinates outside image bounds; originals were preserved. "
            "Review {} before point-based training.".format(
                len(coordinate_errors), report_path
            )
        )
    if any(metadata_warnings.values()):
        print("Annotation metadata warnings: {}".format(metadata_warnings))
    print("Counts and point-list manifests written under {}.".format(dataset_root))
    print("Validation report: {}".format(report_path))
    if coordinate_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
