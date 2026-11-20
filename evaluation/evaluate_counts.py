import argparse
import csv
import sys

csv.field_size_limit(sys.maxsize)
import json
import math
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = MODEL_ROOT / "dataset" / "crowd_counting"
DEFAULT_PREDICTIONS = (
    MODEL_ROOT / "outputs" / "predictions" / "crowd_point_annotations_predictions.csv"
)
DEFAULT_OUTPUT_DIR = MODEL_ROOT / "outputs" / "evaluation"
DEFAULT_HISTORY = MODEL_ROOT / "outputs" / "training" / "history.csv"
DEFAULT_CHECKPOINT = MODEL_ROOT / "original_repo" / "weights" / "SHTechA.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate P2PNet counts against point-annotation counts."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--architecture",
        default="P2PNet",
        help="Architecture label written to the evaluation summary.",
    )
    parser.add_argument(
        "--training-blocker",
        default=None,
        help="Optional architecture-specific reason training was not run.",
    )
    parser.add_argument(
        "--pretrained-weights-used",
        action="store_true",
        help="Set only when these predictions were actually produced with pretrained weights.",
    )
    parser.add_argument(
        "--inference-error",
        default=None,
        help="Record a real inference failure and produce a status report without metrics.",
    )
    parser.add_argument(
        "--training-error",
        default=None,
        help="Record a real training failure without inventing loss values.",
    )
    return parser.parse_args()



def frame_key(name):
    path = PurePosixPath(name.replace("\\", "/"))
    return path.stem



def read_ground_truth(path):
    if not path.is_file():
        raise FileNotFoundError("Counts CSV does not exist: {}".format(path))

    result = {}

    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)

        required = {"frame", "count", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Ground-truth CSV must contain frame,count,split columns."
            )

        for row in reader:
            # Evaluate only the official test split.
            if (row.get("split") or "").strip().lower() != "test":
                continue

            name = (row.get("frame") or "").strip()
            if not name:
                raise ValueError("Ground-truth CSV contains an empty frame name.")

            key = frame_key(name)

            if key in result:
                raise ValueError("Duplicate test ground-truth frame: {}".format(name))

            try:
                count = int(row.get("count", ""))
            except (TypeError, ValueError):
                raise ValueError("Invalid ground-truth count for {}".format(name))

            if count < 0:
                raise ValueError("Negative ground-truth count for {}".format(name))

            result[key] = (name, count)

    if not result:
        raise ValueError("No test-split ground-truth records found.")

    return result



def read_predictions(path):
    if not path.is_file():
        raise FileNotFoundError("Prediction CSV does not exist: {}".format(path))
    result = {}
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"Frame_ID", "Predicted_Count", "Inference_Time_ms"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Prediction CSV must contain Frame_ID,Predicted_Count,"
                "Inference_Time_ms columns."
            )
        for row in reader:
            name = (row.get("Frame_ID") or "").strip()
            if not name:
                raise ValueError("Prediction CSV contains an empty Frame_ID.")
            key = frame_key(name)
            if key in result:
                raise ValueError("Duplicate prediction frame: {}".format(name))
            try:
                count = float(row.get("Predicted_Count", ""))
                latency = float(row.get("Inference_Time_ms", ""))
            except (TypeError, ValueError):
                raise ValueError("Invalid prediction count or latency for {}".format(name))
            if not math.isfinite(count) or not math.isfinite(latency) or latency <= 0:
                raise ValueError(
                    "Prediction count/latency must be finite and latency positive for {}."
                    .format(name)
                )
            result[key] = (count, latency)
    return result


def read_training_history(path):
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"epoch", "training_loss", "validation_loss"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Training history must contain epoch,training_loss,validation_loss."
            )
        history = []
        for row in reader:
            try:
                epoch = int(row["epoch"])
                train_loss = float(row["training_loss"])
                val_loss = float(row["validation_loss"])
            except (TypeError, ValueError):
                raise ValueError("Training history contains invalid loss values.")
            if not all(math.isfinite(value) for value in (train_loss, val_loss)):
                raise ValueError("Training history contains non-finite loss values.")
            history.append((epoch, train_loss, val_loss))
    return history


def save_placeholder_plot(path, title, message):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(str(path), dpi=150)
    plt.close(figure)


def save_training_plots(output_dir, history):
    if not history:
        message = "Training loss history is unavailable."
        save_placeholder_plot(output_dir / "training_loss.png", "Training loss", message)
        save_placeholder_plot(
            output_dir / "validation_loss.png", "Validation loss", message
        )
        return

    epochs = [item[0] for item in history]
    train_losses = [item[1] for item in history]
    val_losses = [item[2] for item in history]
    for filename, title, losses in (
        ("training_loss.png", "Training loss", train_losses),
        ("validation_loss.png", "Validation loss", val_losses),
    ):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(epochs, losses)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(str(output_dir / filename), dpi=150)
        plt.close(figure)


def main():
    args = parse_args()
    ground_truth = read_ground_truth(args.dataset_root / "annotations" / "counts.csv")
    history = read_training_history(args.training_history)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "evaluation_results.csv"
    summary_path = args.output_dir / "performance_summary.json"
    actual_vs_predicted_path = args.output_dir / "actual_vs_predicted.png"
    annotated_dir = args.output_dir / "annotated_predictions"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "architecture": args.architecture,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_size": len(ground_truth),
        "pretrained_weights_available": args.checkpoint.is_file(),
        "pretrained_weights_used": False,
        "training_status": (
            "completed" if history else "failed" if args.training_error else "not_run"
        ),
        "final_training_loss": history[-1][1] if history else None,
        "final_validation_loss": history[-1][2] if history else None,
        "training_error": args.training_error,
        "training_blocker": (
            args.training_blocker
            or (
                "The original P2PNet trainer requires CUDA, but this environment uses "
                "CPU-only PyTorch and no CUDA-capable GPU is available."
                if args.architecture == "P2PNet"
                else None
            )
            if not history
            else None
        ),
        "inference_status": "not_run",
        "frames_evaluated": 0,
        "mae": None,
        "rmse": None,
        "mape_percent": None,
        "mape_frames": 0,
        "zero_ground_truth_frames": sum(
            1 for _, count in ground_truth.values() if count == 0
        ),
        "average_inference_latency_ms": None,
        "processing_fps": None,
        "note": (
            "Point annotations are preliminary, may omit distant/occluded people, "
            "and the 16 sequential frames are not an independent validation set."
        ),
        "evaluation_limitations": [
            "The 16 annotated frames are sequential samples from one clip, not an independent test set.",
            "Preliminary point annotations may omit distant or occluded people.",
            "Predictions and model metrics are unavailable when inference fails.",
        ],
    }

    save_training_plots(args.output_dir, history)
    if args.inference_error:
        summary["inference_status"] = "failed"
        summary["inference_error"] = args.inference_error
        with report_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=[
                    "frame_name",
                    "ground_truth_count",
                    "predicted_count",
                    "absolute_error",
                    "inference_time_ms",
                ],
            )
            writer.writeheader()
        save_placeholder_plot(
            actual_vs_predicted_path,
            "Actual vs predicted counts",
            "Inference failed; no predictions or count metrics are available.",
        )
    else:
        predictions = read_predictions(args.predictions)
        missing = sorted(set(ground_truth) - set(predictions))
        extra = sorted(set(predictions) - set(ground_truth))
        if missing or extra:
            details = []
            if missing:
                details.append("missing predictions for: {}".format(", ".join(missing)))
            if extra:
                details.append(
                    "predictions without ground truth: {}".format(", ".join(extra))
                )
            raise ValueError("Frame sets do not match; {}".format("; ".join(details)))
        if not ground_truth:
            raise ValueError("No frames found to evaluate.")

        per_frame = []
        signed_errors = []
        percentage_errors = []
        latencies = []
        for key in sorted(ground_truth):
            frame_name, actual = ground_truth[key]
            predicted, latency = predictions[key]
            signed_error = predicted - actual
            signed_errors.append(signed_error)
            latencies.append(latency)
            if actual != 0:
                percentage_errors.append(abs(signed_error) / float(actual) * 100.0)
            per_frame.append({
                "frame_name": frame_name,
                "ground_truth_count": actual,
                "predicted_count": "{:.6f}".format(predicted),
                "absolute_error": "{:.6f}".format(abs(signed_error)),
                "inference_time_ms": "{:.3f}".format(latency),
            })

        with report_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=[
                    "frame_name",
                    "ground_truth_count",
                    "predicted_count",
                    "absolute_error",
                    "inference_time_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(per_frame)

        mae = sum(abs(error) for error in signed_errors) / len(signed_errors)
        rmse = math.sqrt(
            sum(error * error for error in signed_errors) / len(signed_errors)
        )
        average_latency = sum(latencies) / len(latencies)
        summary.update({
            "inference_status": "completed",
            "pretrained_weights_used": args.pretrained_weights_used,
            "frames_evaluated": len(per_frame),
            "mae": mae,
            "rmse": rmse,
            "mape_percent": (
                sum(percentage_errors) / len(percentage_errors)
                if percentage_errors
                else None
            ),
            "mape_frames": len(percentage_errors),
            "average_inference_latency_ms": average_latency,
            "processing_fps": 1000.0 / average_latency,
        })

        figure, axis = plt.subplots(figsize=(8, 6))
        actual_values = [ground_truth[key][1] for key in sorted(ground_truth)]
        predicted_values = [predictions[key][0] for key in sorted(ground_truth)]
        axis.scatter(actual_values, predicted_values)
        low = min(actual_values + predicted_values)
        high = max(actual_values + predicted_values)
        axis.plot([low, high], [low, high], linestyle="--", color="gray")
        axis.set_xlabel("Ground-truth count")
        axis.set_ylabel("Predicted count")
        axis.set_title("Actual vs predicted counts")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(str(actual_vs_predicted_path), dpi=150)
        plt.close(figure)

    with summary_path.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2, allow_nan=False)
        destination.write("\n")

    print("Inference status: {}".format(summary["inference_status"]))
    print("Training status: {}".format(summary["training_status"]))
    if summary["inference_status"] == "completed":
        print("MAE: {:.6f}".format(summary["mae"]))
        print("RMSE: {:.6f}".format(summary["rmse"]))
        if summary["mape_percent"] is None:
            print("MAPE: unavailable (all ground-truth counts are zero)")
        else:
            print("MAPE: {:.6f}%".format(summary["mape_percent"]))
        print(
            "Average inference latency: {:.3f} ms; FPS: {:.3f}".format(
                summary["average_inference_latency_ms"], summary["processing_fps"]
            )
        )
    else:
        print("Inference error: {}".format(summary.get("inference_error", "not run")))
    print("CSV: {}".format(report_path))
    print("Summary: {}".format(summary_path))
    print("Plots: {}".format(args.output_dir))
    print("Annotated predictions: {}".format(annotated_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
