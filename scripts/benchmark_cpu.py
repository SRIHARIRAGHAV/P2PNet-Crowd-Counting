import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


MODEL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = MODEL_ROOT / "original_repo"
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(MODEL_ROOT))

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare P2PNet CPU inference configurations on identical frames."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        type=Path,
        default=MODEL_ROOT / "dataset" / "video" / "park_video.mp4",
        help="Video input; frames are decoded once and reused for every configuration.",
    )
    source.add_argument(
        "--frames-dir",
        type=Path,
        help="Image directory; supported formats are BMP, JPEG, PNG, and TIFF.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY / "weights" / "SHTechA.pth",
    )
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--threads",
        default="1,2,4",
        help="Comma-separated PyTorch/optional-runtime thread counts.",
    )
    parser.add_argument("--max-long-edge", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        help="Optional CSV with frame_id,count columns for MAE/RMSE.",
    )
    parser.add_argument(
        "--max-count-difference",
        type=float,
        default=0.0,
        help="Maximum allowed per-frame count difference from original PyTorch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_ROOT / "outputs" / "cpu_benchmark",
    )
    parser.add_argument(
        "--no-optional-backends",
        action="store_true",
        help="Do not attempt ONNX Runtime, OpenVINO, or dynamic INT8.",
    )
    args = parser.parse_args(argv)
    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1.")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative.")
    if args.max_long_edge < 128:
        parser.error("--max-long-edge must be at least 128.")
    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")
    if (
        not math.isfinite(args.max_count_difference)
        or args.max_count_difference < 0
    ):
        parser.error("--max-count-difference must be non-negative.")
    try:
        args.thread_counts = sorted({
            int(value.strip()) for value in args.threads.split(",")
        })
    except ValueError:
        parser.error("--threads must be a comma-separated list of integers.")
    if not args.thread_counts or args.thread_counts[0] < 1:
        parser.error("Each thread count must be at least 1.")
    if any(count > (os.cpu_count() or 1) for count in args.thread_counts):
        parser.error("--threads cannot exceed the number of logical processors.")
    return args


def _resize_dimensions(width, height, max_long_edge):
    scale = min(1.0, float(max_long_edge) / max(width, height))
    resized_width = max(128, int(width * scale))
    resized_height = max(128, int(height * scale))
    return (
        max(128, (resized_width // 128) * 128),
        max(128, (resized_height // 128) * 128),
    )


def _prepare_image(frame, max_long_edge, legacy):
    started = time.perf_counter()
    original_height, original_width = frame.shape[:2]
    original_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_width, resized_height = _resize_dimensions(
        original_width,
        original_height,
        max_long_edge,
    )
    if original_rgb.size != (resized_width, resized_height):
        prepared = original_rgb.resize(
            (resized_width, resized_height),
            Image.BILINEAR,
        )
    else:
        prepared = original_rgb

    if legacy:
        array = np.asarray(prepared, dtype=np.float32).copy().transpose((2, 0, 1))
        tensor = torch.from_numpy(array).div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
    else:
        from inference.run_frames import image_tensor

        tensor = image_tensor(prepared)

    return (
        tensor.unsqueeze(0),
        (original_width, original_height),
        (resized_width, resized_height),
        (time.perf_counter() - started) * 1000.0,
    )


def _run_torch_frame(
    frame,
    model,
    threshold,
    max_long_edge,
    legacy,
    channels_last,
):
    samples, original_size, resized_size, preprocessing_ms = _prepare_image(
        frame,
        max_long_edge,
        legacy,
    )
    if channels_last:
        layout_started = time.perf_counter()
        samples = samples.contiguous(memory_format=torch.channels_last)
        preprocessing_ms += (time.perf_counter() - layout_started) * 1000.0

    inference_started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(samples)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0

    postprocessing_started = time.perf_counter()
    from predict_video import valid_prediction_points

    points = valid_prediction_points(
        outputs,
        threshold,
        original_size,
        resized_size,
    )
    postprocessing_ms = (time.perf_counter() - postprocessing_started) * 1000.0
    return len(points), preprocessing_ms, inference_ms, postprocessing_ms


def _read_frames(args):
    frames = []
    frame_ids = []
    annotation_counts = {}
    if args.frames_dir is not None:
        if not args.frames_dir.is_dir():
            raise FileNotFoundError(
                f"Frames directory does not exist: {args.frames_dir}"
            )
        suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
        paths = sorted(
            path for path in args.frames_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )[:args.max_frames]
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"OpenCV could not decode image '{path}'.")
            frame_id = path.relative_to(args.frames_dir).with_suffix("").as_posix()
            frames.append(frame)
            frame_ids.append(frame_id)
            annotation_path = path.with_suffix(".txt")
            if annotation_path.is_file():
                with annotation_path.open("r", encoding="utf-8") as annotation_file:
                    annotation_counts[frame_id] = sum(
                        bool(line.strip()) for line in annotation_file
                    )
        if not frames:
            raise FileNotFoundError(
                f"No supported images found in {args.frames_dir}."
            )
        return frames, frame_ids, annotation_counts

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open input video '{input_path}'.")
    try:
        while len(frames) < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            frame_ids.append(str(len(frames)))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No video frames could be decoded from '{input_path}'.")
    return frames, frame_ids, annotation_counts


def _read_ground_truth(args, frame_ids, paired_annotations):
    counts = dict(paired_annotations)
    if args.ground_truth_csv is None:
        return counts
    with args.ground_truth_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"frame_id", "count"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "--ground-truth-csv must contain 'frame_id' and 'count' columns."
            )
        for row in reader:
            try:
                count = float(row["count"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Ground-truth CSV count values must be numeric."
                ) from error
            if not math.isfinite(count) or count < 0:
                raise ValueError(
                    "Ground-truth CSV counts must be finite and non-negative."
                )
            counts[row["frame_id"]] = count
    return {frame_id: counts[frame_id] for frame_id in frame_ids if frame_id in counts}


def _count_metrics(predictions, frame_ids, ground_truth):
    pairs = [
        (predictions[index], ground_truth[frame_id])
        for index, frame_id in enumerate(frame_ids)
        if frame_id in ground_truth
    ]
    if not pairs:
        return {"mae": None, "rmse": None, "ground_truth_frames": 0}
    differences = [predicted - actual for predicted, actual in pairs]
    return {
        "mae": statistics.mean(abs(value) for value in differences),
        "rmse": math.sqrt(statistics.mean(value * value for value in differences)),
        "ground_truth_frames": len(pairs),
    }


def _measure_variant(label, backend, threads, frames, frame_ids, ground_truth,
                     runner, warmup, baseline_counts=None, channels_last=False):
    from inference.performance import ProcessResourceMonitor

    for index in range(warmup):
        runner(frames[index % len(frames)])

    resource_monitor = ProcessResourceMonitor()
    cpu_samples = []
    memory_peak = 0.0
    memory_current = 0.0
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []
    total_times = []
    counts = []
    for frame in frames:
        frame_started = time.perf_counter()
        count, preprocessing_ms, inference_ms, postprocessing_ms = runner(frame)
        total_ms = (time.perf_counter() - frame_started) * 1000.0
        counts.append(count)
        preprocessing_times.append(preprocessing_ms)
        inference_times.append(inference_ms)
        postprocessing_times.append(postprocessing_ms)
        total_times.append(total_ms)
        resources = resource_monitor.sample()
        cpu_samples.append(resources["cpu_percent"])
        memory_current = resources["memory_mb"]
        memory_peak = max(memory_peak, resources["peak_memory_mb"])

    count_differences = (
        [abs(count - baseline_counts[index]) for index, count in enumerate(counts)]
        if baseline_counts is not None
        else [0 for _ in counts]
    )
    count_metrics = _count_metrics(counts, frame_ids, ground_truth)
    processing_seconds = sum(total_times) / 1000.0
    total_inference_seconds = sum(inference_times) / 1000.0
    return {
        "configuration": label,
        "backend": backend,
        "threads": threads,
        "channels_last": bool(channels_last),
        "frames": len(frames),
        "average_fps": (
            len(frames) / processing_seconds if processing_seconds > 0 else 0.0
        ),
        "average_inference_fps": (
            len(frames) / total_inference_seconds
            if total_inference_seconds > 0
            else 0.0
        ),
        "average_latency_ms": statistics.mean(total_times),
        "average_inference_latency_ms": statistics.mean(inference_times),
        "average_preprocessing_ms": statistics.mean(preprocessing_times),
        "average_postprocessing_ms": statistics.mean(postprocessing_times),
        "cpu_usage_percent": statistics.mean(cpu_samples),
        "memory_mb": memory_current,
        "peak_memory_mb": memory_peak,
        "mean_absolute_count_difference": statistics.mean(count_differences),
        "maximum_count_difference": max(count_differences),
        **count_metrics,
        "status": "measured",
        "error": None,
    }, counts


def _error_row(label, backend, error, threads=None):
    return {
        "configuration": label,
        "backend": backend,
        "threads": threads,
        "channels_last": False,
        "frames": 0,
        "average_fps": None,
        "average_inference_fps": None,
        "average_latency_ms": None,
        "average_inference_latency_ms": None,
        "average_preprocessing_ms": None,
        "average_postprocessing_ms": None,
        "cpu_usage_percent": None,
        "memory_mb": None,
        "peak_memory_mb": None,
        "mean_absolute_count_difference": None,
        "maximum_count_difference": None,
        "mae": None,
        "rmse": None,
        "ground_truth_frames": 0,
        "status": "unavailable",
        "error": str(error)[:500],
    }


def _run_onnx_backend(model, frames, threshold, max_long_edge, threads,
                      ground_truth, frame_ids, baseline_counts, directory):
    label = f"onnxruntime-cpu-t{threads}"
    resized_shapes = {
        _resize_dimensions(frame.shape[1], frame.shape[0], max_long_edge)
        for frame in frames
    }
    if len(resized_shapes) != 1:
        return [_error_row(
            label,
            "onnxruntime",
            "Static ONNX export requires all benchmark frames to have one resized shape.",
            threads,
        )]
    if importlib.util.find_spec("onnx") is None:
        return [_error_row(label, "onnxruntime", "onnx is not installed.", threads)]
    try:
        model = model.to(memory_format=torch.contiguous_format)
        example, _, _, _ = _prepare_image(frames[0], max_long_edge, legacy=False)
        graph_path = directory / "p2pnet_cpu_benchmark.onnx"
        torch.onnx.export(
            model,
            example,
            str(graph_path),
            input_names=["images"],
            output_names=["pred_logits", "pred_points"],
            opset_version=17,
            do_constant_folding=True,
        )
        if importlib.util.find_spec("onnxruntime") is None:
            return [_error_row(
                label,
                "onnxruntime",
                "onnxruntime is not installed; ONNX export completed for OpenVINO.",
            )]
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(graph_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        def runner(frame):
            samples, original_size, resized_size, preprocessing_ms = _prepare_image(
                frame,
                max_long_edge,
                legacy=False,
            )
            inference_started = time.perf_counter()
            logits, points = session.run(
                ["pred_logits", "pred_points"],
                {"images": samples.numpy()},
            )
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            post_started = time.perf_counter()
            from predict_video import valid_prediction_points

            selected = valid_prediction_points(
                {
                    "pred_logits": torch.from_numpy(logits),
                    "pred_points": torch.from_numpy(points),
                },
                threshold,
                original_size,
                resized_size,
            )
            return (
                len(selected),
                preprocessing_ms,
                inference_ms,
                (time.perf_counter() - post_started) * 1000.0,
            )

        row, _ = _measure_variant(
            label,
            "onnxruntime",
            threads,
            frames,
            frame_ids,
            ground_truth,
            runner,
            warmup=1,
            baseline_counts=baseline_counts,
        )
        return [row]
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        return [_error_row(label, "onnxruntime", error, threads)]


def _run_openvino_backend(frames, threshold, max_long_edge, threads,
                          ground_truth, frame_ids, baseline_counts, directory):
    label = f"openvino-cpu-t{threads}"
    if importlib.util.find_spec("openvino") is None:
        return [_error_row(
            label,
            "openvino",
            "OpenVINO is not installed.",
            threads,
        )]
    if not (directory / "p2pnet_cpu_benchmark.onnx").is_file():
        return [_error_row(
            label,
            "openvino",
            "ONNX export is unavailable.",
            threads,
        )]
    try:
        import openvino as ov

        core = ov.Core()
        ov_model = core.read_model(str(directory / "p2pnet_cpu_benchmark.onnx"))
        compiled = core.compile_model(
            ov_model,
            "CPU",
            {"INFERENCE_NUM_THREADS": threads},
        )

        def runner(frame):
            samples, original_size, resized_size, preprocessing_ms = _prepare_image(
                frame,
                max_long_edge,
                legacy=False,
            )
            inference_started = time.perf_counter()
            results = compiled([samples.numpy()])
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            post_started = time.perf_counter()
            from predict_video import valid_prediction_points

            logits = np.asarray(results[compiled.output("pred_logits")])
            points = np.asarray(results[compiled.output("pred_points")])
            selected = valid_prediction_points(
                {
                    "pred_logits": torch.from_numpy(logits),
                    "pred_points": torch.from_numpy(points),
                },
                threshold,
                original_size,
                resized_size,
            )
            return (
                len(selected),
                preprocessing_ms,
                inference_ms,
                (time.perf_counter() - post_started) * 1000.0,
            )

        row, _ = _measure_variant(
            label,
            "openvino",
            threads,
            frames,
            frame_ids,
            ground_truth,
            runner,
            warmup=1,
            baseline_counts=baseline_counts,
        )
        return [row]
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        return [_error_row(label, "openvino", error, threads)]


def _write_results(rows, output_dir, environment):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cpu_benchmark.json"
    csv_path = output_dir / "cpu_benchmark.csv"
    with json_path.open("w", encoding="utf-8") as result_file:
        json.dump({"environment": environment, "results": rows}, result_file, indent=2)
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main(argv=None):
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    frames, frame_ids, paired_annotations = _read_frames(args)
    ground_truth = _read_ground_truth(args, frame_ids, paired_annotations)

    torch.set_num_interop_threads(1)
    torch.set_num_threads(args.thread_counts[0])
    from models import build_model

    model_args = type("ModelArgs", (), {
        "backbone": "vgg16_bn",
        "row": 2,
        "line": 2,
    })()
    model = build_model(model_args, training=False)
    checkpoint = torch.load(
        str(args.checkpoint),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must contain a state-dict mapping.")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device("cpu"))
    model.eval()

    result_rows = []
    baseline_by_threads = {}
    for threads in args.thread_counts:
        torch.set_num_threads(threads)
        model.to(memory_format=torch.contiguous_format)

        def original_runner(frame):
            return _run_torch_frame(
                frame,
                model,
                args.threshold,
                args.max_long_edge,
                legacy=True,
                channels_last=False,
            )

        original_row, original_counts = _measure_variant(
            f"pytorch-original-t{threads}",
            "pytorch",
            threads,
            frames,
            frame_ids,
            ground_truth,
            original_runner,
            args.warmup,
        )
        result_rows.append(original_row)
        baseline_by_threads[threads] = original_counts

        model.to(memory_format=torch.contiguous_format)

        def optimized_runner(frame):
            return _run_torch_frame(
                frame,
                model,
                args.threshold,
                args.max_long_edge,
                legacy=False,
                channels_last=False,
            )

        optimized_row, _ = _measure_variant(
            f"pytorch-optimized-t{threads}",
            "pytorch",
            threads,
            frames,
            frame_ids,
            ground_truth,
            optimized_runner,
            args.warmup,
            baseline_counts=original_counts,
        )
        result_rows.append(optimized_row)

        model.to(memory_format=torch.channels_last)

        def channels_last_runner(frame):
            return _run_torch_frame(
                frame,
                model,
                args.threshold,
                args.max_long_edge,
                legacy=False,
                channels_last=True,
            )

        channels_row, _ = _measure_variant(
            f"pytorch-channels-last-t{threads}",
            "pytorch",
            threads,
            frames,
            frame_ids,
            ground_truth,
            channels_last_runner,
            args.warmup,
            baseline_counts=original_counts,
            channels_last=True,
        )
        result_rows.append(channels_row)

    if not args.no_optional_backends:
        with tempfile.TemporaryDirectory(prefix="p2pnet-benchmark-") as temp_dir:
            directory = Path(temp_dir)
            for threads in args.thread_counts:
                result_rows.extend(_run_onnx_backend(
                    model,
                    frames,
                    args.threshold,
                    args.max_long_edge,
                    threads,
                    ground_truth,
                    frame_ids,
                    baseline_by_threads[threads],
                    directory,
                ))
                result_rows.extend(_run_openvino_backend(
                    frames,
                    args.threshold,
                    args.max_long_edge,
                    threads,
                    ground_truth,
                    frame_ids,
                    baseline_by_threads[threads],
                    directory,
                ))

    linear_layers = sum(
        isinstance(module, torch.nn.Linear) for module in model.modules()
    )
    if args.no_optional_backends:
        result_rows.append(_error_row(
            "dynamic-int8",
            "pytorch-dynamic-int8",
            "Skipped by --no-optional-backends.",
        ))
    elif linear_layers == 0:
        result_rows.append(_error_row(
            "dynamic-int8",
            "pytorch-dynamic-int8",
            "Not applicable: P2PNet has no nn.Linear layers for dynamic INT8.",
        ))
        result_rows[-1]["status"] = "not_applicable"
    else:
        result_rows.append(_error_row(
            "dynamic-int8",
            "pytorch-dynamic-int8",
            "Dynamic INT8 was not run because the model has no validated quantization path.",
        ))

    environment = {
        "cpu": platform.processor() or platform.machine(),
        "logical_processors": os.cpu_count(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "torch_mkldnn_available": torch.backends.mkldnn.is_available(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "input_frames": len(frames),
        "checkpoint": str(args.checkpoint.resolve()),
        "max_long_edge": args.max_long_edge,
        "confidence_threshold": args.threshold,
        "warmup_iterations": args.warmup,
        "thread_counts": args.thread_counts,
        "ground_truth_frames": len(ground_truth),
    }
    json_path, csv_path = _write_results(
        result_rows,
        args.output_dir,
        environment,
    )
    print(
        "Configuration | Avg FPS | Inference FPS | Latency ms | CPU % | "
        "Memory MiB (peak) | Count diff | MAE | RMSE | Status"
    )
    for row in result_rows:
        print(
            "{configuration} | {fps} | {inference_fps} | {latency} | {cpu} | "
            "{memory} | {difference} | {mae} | {rmse} | {status}".format(
                configuration=row["configuration"],
                fps=(
                    f"{row['average_fps']:.3f}"
                    if row["average_fps"] is not None else "n/a"
                ),
                inference_fps=(
                    f"{row['average_inference_fps']:.3f}"
                    if row["average_inference_fps"] is not None else "n/a"
                ),
                latency=(
                    f"{row['average_latency_ms']:.2f}"
                    if row["average_latency_ms"] is not None else "n/a"
                ),
                cpu=(
                    f"{row['cpu_usage_percent']:.1f}"
                    if row["cpu_usage_percent"] is not None else "n/a"
                ),
                memory=(
                    f"{row['memory_mb']:.1f} ({row['peak_memory_mb']:.1f})"
                    if row["memory_mb"] is not None
                    and row["peak_memory_mb"] is not None else "n/a"
                ),
                difference=(
                    f"{row['mean_absolute_count_difference']:.3f}/"
                    f"{row['maximum_count_difference']}"
                    if row["mean_absolute_count_difference"] is not None else "n/a"
                ),
                mae=f"{row['mae']:.3f}" if row["mae"] is not None else "n/a",
                rmse=f"{row['rmse']:.3f}" if row["rmse"] is not None else "n/a",
                status=row["status"],
            )
        )
    eligible = [
        row for row in result_rows
        if row["status"] == "measured"
        and row["maximum_count_difference"] <= args.max_count_difference
    ]
    if eligible:
        fastest = max(eligible, key=lambda row: row["average_fps"])
        print(
            "Fastest configuration within the count-difference limit: "
            f"{fastest['configuration']} ({fastest['average_fps']:.3f} FPS)."
        )
    else:
        print("No measured configuration met the count-difference limit.")
    print(f"JSON results: {json_path}")
    print(f"CSV results: {csv_path}")


if __name__ == "__main__":
    main()
