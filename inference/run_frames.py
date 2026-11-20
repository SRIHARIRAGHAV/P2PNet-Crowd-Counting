import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image


MODEL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = MODEL_ROOT / "original_repo"
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(MODEL_ROOT))

from inference.device import resolve_device
from models import build_model


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_tensor(image):
    array = np.asarray(image, dtype=np.float32).copy().transpose((2, 0, 1))
    tensor = torch.from_numpy(array).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def parse_args():
    parser = argparse.ArgumentParser(description="Run the existing P2PNet checkpoint on image frames.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=MODEL_ROOT / "dataset" / "crowd_counting" / "images" / "annotated",
    )
    parser.add_argument("--weight", type=Path, default=REPOSITORY / "weights" / "SHTechA.pth")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=MODEL_ROOT / "outputs" / "predictions" / "crowd_point_annotations_predictions.csv",
    )
    parser.add_argument(
        "--visualizations-dir",
        type=Path,
        default=MODEL_ROOT / "outputs" / "evaluation" / "annotated_predictions",
    )
    parser.add_argument("--max-long-edge", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "cuda:0"),
        default="auto",
        help="Inference device (default: CUDA when available, otherwise CPU).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.frames_dir.is_dir():
        raise FileNotFoundError("Frames directory does not exist: {}".format(args.frames_dir))
    if not args.weight.is_file():
        raise FileNotFoundError("P2PNet checkpoint does not exist: {}".format(args.weight))
    if args.max_long_edge < 128:
        raise ValueError("--max-long-edge must be at least 128.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    image_paths = sorted(
        path for path in args.frames_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError("No supported image frames found in {}".format(args.frames_dir))

    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    print("Selected inference device: {}".format(device))
    model = build_model(SimpleNamespace(backbone="vgg16_bn", row=2, line=2), training=False)
    checkpoint = torch.load(
        str(args.weight),
        map_location="cpu",
        weights_only=True,
    )
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.visualizations_dir.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "Frame_ID",
                "Predicted_Count",
                "Inference_Time_ms",
                "Predicted_Points",
            ],
        )
        writer.writeheader()
        for path in image_paths:
            frame_id = path.relative_to(args.frames_dir).with_suffix("").as_posix()
            with Image.open(str(path)) as opened:
                original = opened.convert("RGB")
            original_width, original_height = original.size
            scale = min(1.0, float(args.max_long_edge) / max(original.size))
            resized_width = max(128, int(original_width * scale))
            resized_height = max(128, int(original_height * scale))
            resized_width = max(128, (resized_width // 128) * 128)
            resized_height = max(128, (resized_height // 128) * 128)
            if original.size != (resized_width, resized_height):
                prepared = original.resize((resized_width, resized_height), Image.BILINEAR)
            else:
                prepared = original

            samples = image_tensor(prepared).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(samples)
                scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
                predicted_points = outputs["pred_points"][0][scores > args.threshold]
                predicted_count = int((scores > args.threshold).sum().item())
                points = predicted_points.cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms = (time.perf_counter() - start) * 1000.0

            canvas = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
            x_scale = float(original_width) / resized_width
            y_scale = float(original_height) / resized_height
            original_points = [
                [float(x * x_scale), float(y * y_scale)] for x, y in points
            ]
            for x, y in original_points:
                cv2.circle(
                    canvas,
                    (int(round(x)), int(round(y))),
                    3,
                    (0, 0, 255),
                    -1,
                )
            visualization_path = args.visualizations_dir / path.relative_to(args.frames_dir).with_suffix(".jpg")
            visualization_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(visualization_path), canvas):
                raise OSError("Could not save visualization: {}".format(visualization_path))
            writer.writerow({
                "Frame_ID": frame_id,
                "Predicted_Count": predicted_count,
                "Inference_Time_ms": "{:.3f}".format(inference_ms),
                "Predicted_Points": json.dumps(original_points, separators=(",", ":")),
            })
            print("{}: {} people ({:.1f} ms)".format(frame_id, predicted_count, inference_ms))


if __name__ == "__main__":
    main()
