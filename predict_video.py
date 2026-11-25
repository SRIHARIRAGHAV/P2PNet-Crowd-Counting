import argparse
import csv
import ctypes
import json
import math
import os
import pickle
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import cv2
import numpy as np
import torch
from PIL import Image


MODEL_ROOT = Path(__file__).resolve().parent
REPOSITORY = MODEL_ROOT / "original_repo"
LIVE_WINDOW_NAME = "P2PNet Live Crowd Monitoring"
sys.path.insert(0, str(REPOSITORY))

from inference.device import resolve_device
from inference.run_frames import image_tensor
from inference.performance import (
    ProcessResourceMonitor,
    RollingPerformance,
    RunningStats,
)
from models import build_model


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_argument, _ = config_parser.parse_known_args(argv)
    config_values = {}
    if config_argument.config is not None:
        config_path = config_argument.config
        if not config_path.is_absolute():
            config_path = MODEL_ROOT / config_path
        with config_path.open("r", encoding="utf-8") as config_file:
            config_values = json.load(config_file)
        if not isinstance(config_values, dict):
            raise ValueError("Inference JSON config must contain an object.")

    parser = argparse.ArgumentParser(
        description="Run the existing P2PNet checkpoint on video or RTSP frames."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config; explicit command-line arguments override it.",
    )
    parser.add_argument("--mode", choices=("video", "rtsp"), default="video")
    parser.add_argument(
        "--input",
        type=Path,
        default=MODEL_ROOT / "dataset" / "video" / "park_video.mp4",
        help="Input video (defaults to the project park sample).",
    )
    parser.add_argument("--source", help="RTSP/RTSPS URL; credentials are never logged.")
    parser.add_argument(
        "--source-env",
        default="P2PNET_RTSP_URL",
        help="Environment variable used for the RTSP URL if --source is omitted.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint",
        "--weight",
        dest="checkpoint",
        type=Path,
        default=REPOSITORY / "weights" / "SHTechA.pth",
        help="Existing P2PNet checkpoint; no weights are downloaded or initialized as a fallback.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "cuda:0"),
        default="auto",
        help="Inference device (default: CUDA when available, otherwise CPU).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-long-edge", type=int, default=1024)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="PyTorch CPU intra-op threads; 0 preserves PyTorch's default.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels-last tensors on CPU; benchmark before enabling.",
    )
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--low-max-count", type=int, default=20)
    parser.add_argument("--moderate-max-count", type=int, default=50)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--display-width",
        type=int,
        help="Maximum live-window width in pixels; use with --display-height.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        help="Maximum live-window height in pixels; use with --display-width.",
    )
    parser.add_argument("--reconnect-interval", type=float, default=3.0)
    parser.add_argument("--max-reconnect-attempts", type=int, default=10)
    parser.add_argument("--open-timeout-ms", type=int, default=5000)
    parser.add_argument("--read-timeout-ms", type=int, default=3000)
    parser.add_argument("--recording-fps", type=float, default=15.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for a short smoke test; 0 processes the whole video.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output video and sidecar files if they already exist.",
    )

    actions = {action.dest: action for action in parser._actions}
    allowed_config_keys = set(actions) - {"help", "config"}
    unknown_keys = set(config_values) - allowed_config_keys
    if unknown_keys:
        raise ValueError(
            "Unknown inference config option(s): "
            + ", ".join(sorted(unknown_keys))
        )
    if "source" in config_values:
        raise ValueError(
            "Do not store an RTSP URL in a config file; use --source-env instead."
        )

    boolean_options = {
        "record",
        "no_display",
        "overwrite",
        "channels_last",
    }
    defaults = {}
    for key, value in config_values.items():
        action = actions[key]
        if key in boolean_options:
            if not isinstance(value, bool):
                raise ValueError(
                    "Config option '{}' must be true or false.".format(key)
                )
        elif value is None and key in {
            "output",
            "display_width",
            "display_height",
        }:
            pass
        elif action.type is not None:
            if isinstance(value, bool):
                raise ValueError(
                    "Config option '{}' has an invalid value.".format(key)
                )
            try:
                value = action.type(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Config option '{}' has an invalid value.".format(key)
                ) from error
            if isinstance(value, Path) and not value.is_absolute():
                value = MODEL_ROOT / value
        if action.choices is not None and value not in action.choices:
            raise ValueError(
                "Config option '{}' must be one of {}.".format(
                    key,
                    ", ".join(str(choice) for choice in action.choices),
                )
            )
        defaults[key] = value
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def status_for_count(count, low_max_count, moderate_max_count):
    if count <= low_max_count:
        return "LOW"
    if count <= moderate_max_count:
        return "MODERATE"
    return "HIGH"


def timestamp_for_frame(frame_index, fps):
    total_milliseconds = int(round(frame_index * 1000.0 / fps))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def valid_prediction_points(outputs, threshold, original_size, resized_size):
    scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
    points = outputs["pred_points"][0][scores > threshold].detach().cpu().numpy()
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    points = points.astype(np.float32, copy=True)
    points[:, 0] *= float(original_width) / resized_width
    points[:, 1] *= float(original_height) / resized_height
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < original_width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < original_height)
    )
    return points[valid]


def device_summary(device):
    summary = {
        "inference_device": str(device),
        "cpu_intra_op_threads": torch.get_num_threads(),
        "cpu_inter_op_threads": torch.get_num_interop_threads(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        summary.update({
            "gpu_name": properties.name,
            "gpu_memory_total_mb": properties.total_memory / (1024 ** 2),
            "gpu_memory_allocated_mb": (
                torch.cuda.memory_allocated(device) / (1024 ** 2)
            ),
            "gpu_memory_reserved_mb": (
                torch.cuda.memory_reserved(device) / (1024 ** 2)
            ),
        })
    else:
        summary.update({
            "gpu_name": None,
            "gpu_memory_total_mb": None,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
        })
    return summary


def print_device_summary(device):
    summary = device_summary(device)
    print(f"Selected inference device: {summary['inference_device']}")
    print(
        "PyTorch CPU threads: "
        f"intra-op={summary['cpu_intra_op_threads']}, "
        f"inter-op={summary['cpu_inter_op_threads']}"
    )
    if device.type == "cuda":
        print(
            f"GPU: {summary['gpu_name']}; memory allocated/reserved/total: "
            f"{summary['gpu_memory_allocated_mb']:.1f}/"
            f"{summary['gpu_memory_reserved_mb']:.1f}/"
            f"{summary['gpu_memory_total_mb']:.1f} MiB"
        )
    else:
        print("GPU: unavailable (CPU inference); GPU memory: unavailable")


def print_runtime_summary(checkpoint_path, device):
    cuda_runtime = torch.version.cuda if torch.cuda.is_available() else None
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA availability: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA runtime: {cuda_runtime or 'not available'}")
    if device.type == "cuda":
        print(
            "GPU device: "
            f"{torch.cuda.get_device_name(device)}"
        )
    else:
        print("GPU device: unavailable")
    print(f"Selected inference device: {device}")
    print(f"Checkpoint path: {checkpoint_path}")


def prepare_and_infer(frame, model, threshold, max_long_edge):
    preprocessing_started = time.perf_counter()
    original_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_width, original_height = original_rgb.size
    scale = min(1.0, float(max_long_edge) / max(original_rgb.size))
    resized_width = max(128, int(original_width * scale))
    resized_height = max(128, int(original_height * scale))
    resized_width = max(128, (resized_width // 128) * 128)
    resized_height = max(128, (resized_height // 128) * 128)
    if original_rgb.size != (resized_width, resized_height):
        prepared = original_rgb.resize(
            (resized_width, resized_height),
            Image.BILINEAR,
        )
    else:
        prepared = original_rgb

    device = next(model.parameters()).device
    samples = image_tensor(prepared).unsqueeze(0).to(device)
    if samples.device.type == "cpu" and next(model.parameters()).is_contiguous(
        memory_format=torch.channels_last
    ):
        samples = samples.contiguous(memory_format=torch.channels_last)
    preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(samples)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    postprocessing_started = time.perf_counter()
    points = valid_prediction_points(
        outputs,
        threshold,
        (original_width, original_height),
        (resized_width, resized_height),
    )
    postprocessing_ms = (time.perf_counter() - postprocessing_started) * 1000.0
    return points, inference_ms, preprocessing_ms, postprocessing_ms


def available_display_bounds():
    try:
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return max(1, int(width * 0.9)), max(1, int(height * 0.85))
    except (AttributeError, OSError):
        pass
    return 1280, 720


def automatic_display_size(frame_width, frame_height):
    max_width, max_height = available_display_bounds()
    scale = min(1.0, max_width / frame_width, max_height / frame_height)
    return max(1, round(frame_width * scale)), max(1, round(frame_height * scale))


def fit_frame_to_display(frame, display_size):
    display_width, display_height = display_size
    if display_width < 1 or display_height < 1:
        raise ValueError("Display dimensions must be positive.")
    frame_height, frame_width = frame.shape[:2]
    scale = min(display_width / frame_width, display_height / frame_height)
    resized_width = max(1, min(display_width, round(frame_width * scale)))
    resized_height = max(1, min(display_height, round(frame_height * scale)))
    offset_x = (display_width - resized_width) // 2
    offset_y = (display_height - resized_height) // 2
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    canvas_shape = (display_height, display_width) + frame.shape[2:]
    canvas = np.zeros(canvas_shape, dtype=frame.dtype)
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    transform = (
        resized_width / frame_width,
        resized_height / frame_height,
        offset_x,
        offset_y,
        resized_width,
        resized_height,
    )
    return canvas, transform


def map_points_to_display(points, transform):
    scale_x, scale_y, offset_x, offset_y, resized_width, resized_height = transform
    display_points = np.asarray(points, dtype=np.float32).copy()
    if display_points.size:
        display_points[:, 0] = (
            (display_points[:, 0] + 0.5) * scale_x - 0.5 + offset_x
        )
        display_points[:, 1] = (
            (display_points[:, 1] + 0.5) * scale_y - 0.5 + offset_y
        )
        np.clip(
            display_points[:, 0],
            offset_x,
            offset_x + resized_width - 1,
            out=display_points[:, 0],
        )
        np.clip(
            display_points[:, 1],
            offset_y,
            offset_y + resized_height - 1,
            out=display_points[:, 1],
        )
    return display_points


def live_window_size(window_name, fallback_size):
    if hasattr(cv2, "getWindowImageRect"):
        try:
            _, _, width, height = cv2.getWindowImageRect(window_name)
            if width >= 64 and height >= 64:
                return width, height
        except cv2.error:
            pass
    return fallback_size


def resolve_rtsp_source(args):
    source = args.source
    if not source:
        source = os.environ.get(args.source_env)
    if not source:
        raise ValueError(
            "RTSP mode requires --source or an RTSP URL in the "
            f"{args.source_env} environment variable."
        )
    try:
        parsed = urlsplit(source)
        port = parsed.port
    except ValueError as error:
        raise ValueError("RTSP URL has an invalid port or format.") from error
    if (
        parsed.scheme.lower() not in {"rtsp", "rtsps"}
        or not parsed.hostname
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("RTSP source must be a valid rtsp:// or rtsps:// URL.")
    return source


class LatestFrameReader:
    """Keep only the newest decoded frame; inference never accumulates a backlog."""

    def __init__(
        self,
        source,
        reconnect_interval,
        max_reconnect_attempts,
        open_timeout_ms,
        read_timeout_ms,
        capture_factory=None,
    ):
        self._source = source
        self._reconnect_interval = reconnect_interval
        self._max_reconnect_attempts = max_reconnect_attempts
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._capture_factory = capture_factory or self._open_capture
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="p2pnet-rtsp-reader",
            daemon=True,
        )
        self._frame = None
        self._frame_timestamp = None
        self._sequence = 0
        self._status = "CONNECTING"
        self._finished = False
        self._failure_reason = None
        self._reconnection_attempts = 0
        self._source_fps = 0.0

    def _open_capture(self, source, backend, parameters):
        return cv2.VideoCapture(source, backend, parameters)

    def start(self):
        self._thread.start()

    def get_latest(self, after_sequence, timeout=0.1):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._finished,
                timeout=timeout,
            )
            frame = None
            timestamp = None
            if self._sequence > after_sequence and self._frame is not None:
                frame = self._frame.copy()
                timestamp = self._frame_timestamp
            return (
                self._sequence,
                frame,
                timestamp,
                self._status,
                self._finished,
                self._failure_reason,
            )

    @property
    def reconnection_attempts(self):
        with self._condition:
            return self._reconnection_attempts

    @property
    def source_fps(self):
        with self._condition:
            return self._source_fps

    def stop(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        timeout = (
            self._open_timeout_ms + self._read_timeout_ms
        ) / 1000.0 + self._reconnect_interval + 2.0
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _set_status(self, status):
        with self._condition:
            self._status = status
            self._condition.notify_all()

    def _read_loop(self):
        connection_attempted = False
        try:
            while not self._stop_event.is_set():
                if connection_attempted:
                    with self._condition:
                        if (
                            self._reconnection_attempts
                            >= self._max_reconnect_attempts
                        ):
                            self._status = "DISCONNECTED"
                            self._failure_reason = (
                                "RTSP connection failed after "
                                f"{self._reconnection_attempts} reconnection "
                                "attempts. Check the URL, credentials, camera "
                                "availability, and stream codec."
                            )
                            self._finished = True
                            self._condition.notify_all()
                            return
                        self._reconnection_attempts += 1
                        self._status = "RECONNECTING"
                        self._condition.notify_all()
                    if self._stop_event.wait(self._reconnect_interval):
                        break

                connection_attempted = True
                capture = None
                try:
                    parameters = [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        self._open_timeout_ms,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        self._read_timeout_ms,
                    ]
                    capture = self._capture_factory(
                        self._source,
                        cv2.CAP_FFMPEG,
                        parameters,
                    )
                    if not capture.isOpened():
                        self._set_status("RECONNECTING")
                        continue

                    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
                    with self._condition:
                        if math.isfinite(reported_fps) and reported_fps > 0:
                            self._source_fps = reported_fps
                        self._status = "CONNECTED"
                        self._condition.notify_all()

                    while not self._stop_event.is_set():
                        ok, frame = capture.read()
                        if not ok or frame is None or frame.size == 0:
                            self._set_status("RECONNECTING")
                            break
                        with self._condition:
                            self._sequence += 1
                            self._frame = frame.copy()
                            self._frame_timestamp = time.time()
                            self._status = "CONNECTED"
                            self._condition.notify_all()
                except cv2.error:
                    self._set_status("RECONNECTING")
                finally:
                    if capture is not None:
                        capture.release()
        finally:
            with self._condition:
                if self._stop_event.is_set():
                    self._status = "STOPPED"
                elif not self._finished:
                    self._status = "DISCONNECTED"
                self._finished = True
                self._condition.notify_all()


def draw_head_points(frame, points):
    height, width = frame.shape[:2]
    radius = max(3, min(4, int(round(width / 300.0))))
    for x, y_point in points:
        center = (
            int(np.clip(round(float(x)), 0, width - 1)),
            int(np.clip(round(float(y_point)), 0, height - 1)),
        )
        cv2.circle(frame, center, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, (0, 0, 255), -1, cv2.LINE_AA)


def draw_overlay(
    frame,
    points,
    frame_number,
    timestamp,
    status,
):
    height, width = frame.shape[:2]
    scale = max(0.35, min(0.65, width / 1000.0 * 0.58))
    line_height = max(22, int(30 * scale / 0.58))
    panel_width = min(width, max(320, int(width * 0.48)))
    panel_height = min(height, line_height * 5 + 14)

    panel = frame[:panel_height, :panel_width]
    shade = panel.copy()
    cv2.rectangle(shade, (0, 0), (panel_width, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(shade, 0.72, panel, 0.28, 0, panel)

    margin_x = max(8, int(14 * scale / 0.58))
    y = max(18, int(24 * scale / 0.58))
    lines = [
        ("SAFE LENS AI - CROWD COUNTING", (255, 255, 255), 1),
        (f"CROWD COUNT: {len(points)}", (0, 230, 255), 2),
        (f"FRAME NUMBER: {frame_number}", (255, 255, 255), 1),
        (f"TIMESTAMP: {timestamp}", (255, 255, 255), 1),
        (f"STATUS: {status} (DEMO ONLY)", (120, 255, 120), 1),
    ]
    for text, color, thickness in lines:
        cv2.putText(
            frame,
            text,
            (margin_x, min(y, height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height

    draw_head_points(frame, points)


def draw_live_overlay(
    frame,
    points,
    camera_status,
    frame_number,
    timestamp,
    inference_ms,
    processing_fps,
    crowd_status,
    performance=None,
):
    height, width = frame.shape[:2]
    scale = max(0.35, min(0.65, width / 1000.0 * 0.58))
    line_height = max(22, int(29 * scale / 0.58))
    panel_width = min(width, max(360, int(width * 0.55)))
    status_color = (
        (90, 230, 90)
        if camera_status == "CONNECTED"
        else (0, 180, 255)
        if camera_status in {"CONNECTING", "RECONNECTING"}
        else (80, 80, 255)
    )
    performance = performance or {}
    lines = [
        ("SAFE LENS AI - LIVE CROWD MONITORING", (255, 255, 255), 1),
        (f"CROWD COUNT: {len(points)}", (0, 230, 255), 2),
        (f"CAMERA STATUS: {camera_status}", status_color, 1),
        (f"FRAME NUMBER: {frame_number}", (255, 255, 255), 1),
        (f"STREAM TIMESTAMP: {timestamp}", (255, 255, 255), 1),
        (
            f"INFERENCE: {inference_ms:.1f} ms "
            f"(avg {performance.get('inference_ms', inference_ms):.1f} ms)",
            (255, 255, 255),
            1,
        ),
        (
            f"PRE/POST: {performance.get('preprocessing_ms', 0.0):.1f}/"
            f"{performance.get('postprocessing_ms', 0.0):.1f} ms",
            (255, 255, 255),
            1,
        ),
        (
            f"INFERENCE FPS: {performance.get('inference_fps', 0.0):.2f}",
            (255, 255, 255),
            1,
        ),
        (
            f"PROCESSING FPS: {performance.get('processing_fps', processing_fps):.2f}",
            (255, 255, 255),
            1,
        ),
        (
            f"CPU: {performance.get('cpu_percent', 0.0):.0f}%  "
            f"RAM: {performance.get('memory_mb', 0.0):.0f} MiB",
            (255, 255, 255),
            1,
        ),
        (
            f"PROCESSED FRAMES: {performance.get('processed_frames', 0)}",
            (255, 255, 255),
            1,
        ),
        (f"CROWD STATUS: {crowd_status} (DEMO ONLY)", (120, 255, 120), 1),
    ]
    panel_height = min(height, line_height * len(lines) + 16)
    panel = frame[:panel_height, :panel_width]
    shade = panel.copy()
    cv2.rectangle(shade, (0, 0), (panel_width, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(shade, 0.72, panel, 0.28, 0, panel)

    margin_x = max(8, int(14 * scale / 0.58))
    y = max(18, int(23 * scale / 0.58))
    for text, color, thickness in lines:
        cv2.putText(
            frame,
            text,
            (margin_x, min(y, height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
    draw_head_points(frame, points)


def load_model(checkpoint_path, device):
    model_args = SimpleNamespace(backbone="vgg16_bn", row=2, line=2)
    model = build_model(model_args, training=False)
    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise RuntimeError(
            f"Could not load P2PNet checkpoint '{checkpoint_path}': {error}"
        ) from error

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Unsupported checkpoint format in '{checkpoint_path}': "
            f"expected a state-dict or a mapping containing 'model', "
            f"got {type(checkpoint).__name__}."
        )
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Unsupported model state in '{checkpoint_path}': "
            f"expected a state-dict, got {type(state_dict).__name__}."
        )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is not compatible with the "
            f"existing P2PNet architecture: {error}"
        ) from error
    model.to(device)
    model.eval()
    print(f"Checkpoint loading status: loaded successfully with strict=True")
    return model


def ffmpeg_backend_available():
    registry = getattr(cv2, "videoio_registry", None)
    if registry is None:
        return False
    return registry.hasBackend(cv2.CAP_FFMPEG)


def _temporary_path(path, tag):
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def rtsp_output_paths(output_dir, output_path, overwrite):
    csv_path = output_dir / "rtsp_frame_counts.csv"
    summary_path = output_dir / "rtsp_video_summary.json"
    final_paths = [csv_path, summary_path]
    if output_path is not None:
        final_paths.append(output_path)

    if overwrite or not any(path.exists() for path in final_paths):
        return csv_path, summary_path, output_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = 0
    while True:
        run_tag = timestamp if suffix == 0 else f"{timestamp}_{suffix}"
        run_csv_path = output_dir / f"rtsp_frame_counts_{run_tag}.csv"
        run_summary_path = output_dir / f"rtsp_video_summary_{run_tag}.json"
        run_output_path = (
            output_path.with_name(
                f"{output_path.stem}_{run_tag}{output_path.suffix}"
            )
            if output_path is not None
            else None
        )
        candidates = [run_csv_path, run_summary_path]
        if run_output_path is not None:
            candidates.append(run_output_path)
        if not any(path.exists() for path in candidates):
            return run_csv_path, run_summary_path, run_output_path
        suffix += 1


def run_rtsp(args, model, source, capture_factory=None):
    if not ffmpeg_backend_available():
        raise RuntimeError(
            "RTSP mode requires OpenCV built with the FFmpeg video backend."
        )
    if not hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC") or not hasattr(
        cv2, "CAP_PROP_READ_TIMEOUT_MSEC"
    ):
        raise RuntimeError(
            "This OpenCV version lacks bounded FFmpeg open/read timeouts. "
            "Upgrade OpenCV to use RTSP mode safely."
        )

    output_dir = MODEL_ROOT / "outputs" / "rtsp"
    output_path = None
    if args.record:
        output_path = (
            args.output.resolve()
            if args.output is not None
            else output_dir / "rtsp_crowd_count_output.mp4"
        )
    if args.record and output_path.suffix.lower() != ".mp4":
        raise ValueError("RTSP recording output must use the .mp4 extension.")
    if not args.record and args.output is not None:
        raise ValueError("--output in RTSP mode requires --record.")

    output_dir.mkdir(parents=True, exist_ok=True)
    original_paths = [
        output_dir / "rtsp_frame_counts.csv",
        output_dir / "rtsp_video_summary.json",
    ]
    if output_path is not None:
        original_paths.append(output_path)
    has_existing_results = any(path.exists() for path in original_paths)
    csv_path, summary_path, output_path = rtsp_output_paths(
        output_dir,
        output_path,
        args.overwrite,
    )
    if has_existing_results and not args.overwrite:
        print(
            "Existing RTSP results were preserved. Saving this run to new "
            "timestamped output paths."
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}.{time.time_ns()}.partial"
    temporary_csv = _temporary_path(csv_path, tag)
    temporary_summary = _temporary_path(summary_path, tag)
    temporary_video = (
        _temporary_path(output_path, tag) if output_path is not None else None
    )
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
    )

    reader = LatestFrameReader(
        source,
        args.reconnect_interval,
        args.max_reconnect_attempts,
        args.open_timeout_ms,
        args.read_timeout_ms,
        capture_factory=capture_factory,
    )
    writer = None
    recording_started = False
    recording_size = None
    count_stats = RunningStats()
    inference_stats = RunningStats()
    preprocessing_stats = RunningStats()
    postprocessing_stats = RunningStats()
    processing_stats = RunningStats()
    cpu_stats = RunningStats()
    rolling_performance = RollingPerformance()
    resource_monitor = ProcessResourceMonitor()
    latest_resources = {
        "cpu_percent": 0.0,
        "memory_mb": 0.0,
        "peak_memory_mb": 0.0,
    }
    processed_frames = 0
    last_sequence = 0
    last_frame = None
    last_frame_number = 0
    last_points = np.empty((0, 2), dtype=np.float32)
    last_timestamp = "--"
    last_inference_ms = 0.0
    last_preprocessing_ms = 0.0
    last_postprocessing_ms = 0.0
    last_processing_fps = 0.0
    last_performance = rolling_performance.snapshot()
    last_camera_status = "CONNECTING"
    display_size = None
    display_window_created = False
    failure_reason = None
    started = time.monotonic()
    interrupted = False
    completed = False
    reader.start()
    try:
        with temporary_csv.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "frame_number",
                    "stream_timestamp",
                    "estimated_crowd_count",
                    "predicted_head_point_count",
                    "preprocessing_time_ms",
                    "inference_time_ms",
                    "postprocessing_time_ms",
                    "total_processing_time_ms",
                    "inference_fps",
                    "processing_fps",
                    "cpu_percent",
                    "memory_mb",
                    "camera_connection_status",
                ],
            )
            csv_writer.writeheader()
            try:
                while True:
                    elapsed = time.monotonic() - started
                    if args.duration > 0 and elapsed >= args.duration:
                        break
                    if args.max_frames and processed_frames >= args.max_frames:
                        break

                    (
                        sequence,
                        frame,
                        captured_at,
                        camera_status,
                        reader_finished,
                        failure_reason,
                    ) = reader.get_latest(last_sequence, timeout=0.05)
                    if sequence > last_sequence:
                        last_sequence = sequence
                    last_camera_status = camera_status

                    selected = (
                        frame is not None
                        and (sequence - 1) % args.frame_skip == 0
                    )
                    if selected:
                        processing_started = time.perf_counter()
                        (
                            last_points,
                            last_inference_ms,
                            last_preprocessing_ms,
                            last_postprocessing_ms,
                        ) = prepare_and_infer(
                            frame,
                            model,
                            args.threshold,
                            args.max_long_edge,
                        )
                        frame_number = sequence
                        last_frame_number = frame_number
                        last_timestamp = datetime.fromtimestamp(
                            captured_at
                        ).astimezone().isoformat(timespec="milliseconds")
                        crowd_status = status_for_count(
                            len(last_points),
                            args.low_max_count,
                            args.moderate_max_count,
                        )

                        if output_path is not None:
                            frame_size = (frame.shape[1], frame.shape[0])
                            if recording_size is None:
                                recording_fps = (
                                    reader.source_fps
                                    if reader.source_fps > 0
                                    else args.recording_fps
                                )
                                writer = cv2.VideoWriter(
                                    str(temporary_video),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    recording_fps,
                                    frame_size,
                                )
                                if not writer.isOpened():
                                    writer.release()
                                    writer = None
                                    raise RuntimeError(
                                        f"OpenCV could not create RTSP recording "
                                        f"'{output_path}'. Check the output path "
                                        "and MP4 codec support."
                                    )
                                recording_size = frame_size
                                recording_started = True
                            elif frame_size != recording_size:
                                raise RuntimeError(
                                    "RTSP frame dimensions changed during recording; "
                                    "stopping to avoid a corrupt output video."
                                )

                        last_frame = frame.copy()
                        draw_live_overlay(
                            frame,
                            last_points,
                            camera_status,
                            frame_number,
                            last_timestamp,
                            last_inference_ms,
                            last_processing_fps,
                            crowd_status,
                            {
                                **last_performance,
                                **latest_resources,
                                "processed_frames": processed_frames,
                            },
                        )
                        if writer is not None:
                            writer.write(frame)

                        processing_seconds = max(
                            time.perf_counter() - processing_started,
                            1e-9,
                        )
                        processing_ms = processing_seconds * 1000.0
                        count_stats.add(len(last_points))
                        inference_stats.add(last_inference_ms)
                        preprocessing_stats.add(last_preprocessing_ms)
                        postprocessing_stats.add(last_postprocessing_ms)
                        processing_stats.add(processing_ms)
                        processed_frames += 1
                        rolling_performance.add(
                            last_preprocessing_ms,
                            last_inference_ms,
                            last_postprocessing_ms,
                            processing_ms,
                        )
                        last_performance = rolling_performance.snapshot()
                        latest_resources = resource_monitor.sample()
                        cpu_stats.add(latest_resources["cpu_percent"])
                        last_processing_fps = last_performance["processing_fps"]

                        csv_writer.writerow({
                            "frame_number": frame_number,
                            "stream_timestamp": last_timestamp,
                            "estimated_crowd_count": len(last_points),
                            "predicted_head_point_count": len(last_points),
                            "preprocessing_time_ms": (
                                f"{last_preprocessing_ms:.3f}"
                            ),
                            "inference_time_ms": f"{last_inference_ms:.3f}",
                            "postprocessing_time_ms": (
                                f"{last_postprocessing_ms:.3f}"
                            ),
                            "total_processing_time_ms": f"{processing_ms:.3f}",
                            "inference_fps": (
                                f"{last_performance['inference_fps']:.3f}"
                            ),
                            "processing_fps": f"{last_processing_fps:.3f}",
                            "cpu_percent": f"{latest_resources['cpu_percent']:.1f}",
                            "memory_mb": f"{latest_resources['memory_mb']:.1f}",
                            "camera_connection_status": camera_status,
                        })
                        print(
                            f"Frame {frame_number}: {len(last_points)} people, "
                            f"inference {last_performance['inference_ms']:.1f} ms "
                            f"({last_performance['inference_fps']:.2f} FPS), "
                            f"processing {last_processing_fps:.2f} FPS, "
                            f"CPU {latest_resources['cpu_percent']:.0f}%, "
                            f"RAM {latest_resources['memory_mb']:.0f} MiB"
                        )

                    if not args.no_display and last_frame is not None:
                        if not display_window_created:
                            if args.display_width is None:
                                display_size = automatic_display_size(
                                    last_frame.shape[1],
                                    last_frame.shape[0],
                                )
                            else:
                                display_size = (
                                    args.display_width,
                                    args.display_height,
                                )
                            window_flags = cv2.WINDOW_NORMAL | getattr(
                                cv2,
                                "WINDOW_FREERATIO",
                                0,
                            )
                            cv2.namedWindow(LIVE_WINDOW_NAME, window_flags)
                            cv2.resizeWindow(
                                LIVE_WINDOW_NAME,
                                display_size[0],
                                display_size[1],
                            )
                            display_window_created = True

                        viewport_size = live_window_size(
                            LIVE_WINDOW_NAME,
                            display_size,
                        )
                        display_frame, display_transform = fit_frame_to_display(
                            last_frame,
                            viewport_size,
                        )
                        display_points = map_points_to_display(
                            last_points,
                            display_transform,
                        )
                        draw_live_overlay(
                            display_frame,
                            display_points,
                            camera_status,
                            last_frame_number,
                            last_timestamp,
                            last_inference_ms,
                            last_processing_fps,
                            status_for_count(
                                len(last_points),
                                args.low_max_count,
                                args.moderate_max_count,
                            ),
                            {
                                **last_performance,
                                **latest_resources,
                                "processed_frames": processed_frames,
                            },
                        )
                        cv2.imshow(LIVE_WINDOW_NAME, display_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    if reader_finished:
                        if failure_reason:
                            print(f"RTSP stopped: {failure_reason}")
                        break
            except KeyboardInterrupt:
                interrupted = True
                print("RTSP monitoring interrupted by Ctrl+C.")

        stopped = reader.stop()
        if writer is not None:
            writer.release()
            writer = None
        if not stopped:
            raise RuntimeError(
                "RTSP frame reader did not stop within its configured timeouts."
            )
        if not args.no_display:
            cv2.destroyAllWindows()

        monitoring_duration = time.monotonic() - started
        summary = {
            "source_type": "RTSP",
            "checkpoint": str(args.checkpoint.resolve()),
            **device_summary(next(model.parameters()).device),
            "recording_enabled": output_path is not None,
            "output_video": str(output_path) if output_path is not None else None,
            "csv_results": str(csv_path),
            "total_processed_frames": processed_frames,
            "average_crowd_count": count_stats.mean if count_stats.count else None,
            "maximum_crowd_count": count_stats.maximum,
            "minimum_crowd_count": count_stats.minimum,
            "average_inference_time_ms": inference_stats.mean,
            "average_preprocessing_time_ms": preprocessing_stats.mean,
            "average_postprocessing_time_ms": postprocessing_stats.mean,
            "average_total_processing_time_ms": processing_stats.mean,
            "average_processing_fps": (
                processed_frames * 1000.0 / processing_stats.total
                if processing_stats.total > 0
                else 0.0
            ),
            "average_processed_frame_fps_over_monitoring": (
                processed_frames / monitoring_duration
                if monitoring_duration > 0
                else 0.0
            ),
            "average_inference_fps": (
                processed_frames * 1000.0 / inference_stats.total
                if inference_stats.total > 0
                else 0.0
            ),
            "rolling_window_frames": 30,
            "final_rolling_performance": last_performance,
            "average_cpu_percent": cpu_stats.mean,
            "memory_mb": latest_resources["memory_mb"],
            "peak_memory_mb": latest_resources["peak_memory_mb"],
            "total_stream_monitoring_duration_seconds": monitoring_duration,
            "reconnection_attempts": reader.reconnection_attempts,
            "final_camera_status": last_camera_status,
            "source_fps_reported": reader.source_fps or None,
            "frame_skip": args.frame_skip,
            "confidence_threshold": args.threshold,
            "crowd_status_thresholds": {
                "low": f"0-{args.low_max_count}",
                "moderate": f"{args.low_max_count + 1}-{args.moderate_max_count}",
                "high": f"above {args.moderate_max_count}",
            },
            "crowd_status_note": (
                "Illustrative demonstration thresholds only; not validated "
                "safety thresholds."
            ),
            "timing_note": (
                "Inference time measures the P2PNet forward pass only. Total "
                "processing time per selected frame includes preprocessing, "
                "inference, point filtering, overlay drawing, and optional video "
                "writing; stream monitoring duration includes connection waits."
            ),
            "source_url_redacted": True,
            "interrupted_by_user": interrupted,
        }
        with temporary_summary.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)

        if output_path is not None and recording_started:
            temporary_video.replace(output_path)
        temporary_csv.replace(csv_path)
        temporary_summary.replace(summary_path)
        completed = True
        print(json.dumps(summary, indent=2))
        if output_path is not None and processed_frames:
            print(f"Annotated RTSP recording: {output_path}")
        print(f"RTSP frame counts CSV: {csv_path}")
        print(f"RTSP summary: {summary_path}")

        if failure_reason and processed_frames == 0:
            raise RuntimeError(failure_reason)
    finally:
        if not reader.stop():
            print("Warning: RTSP frame reader did not stop within its timeout.")
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if not completed:
            for path in (temporary_video, temporary_csv, temporary_summary):
                if path is not None and path.exists():
                    path.unlink()


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.max_long_edge < 128:
        raise ValueError("--max-long-edge must be at least 128.")
    if args.threads < 0:
        raise ValueError("--threads cannot be negative.")
    if args.low_max_count < 0 or args.moderate_max_count <= args.low_max_count:
        raise ValueError(
            "Crowd status thresholds must satisfy "
            "0 <= --low-max-count < --moderate-max-count."
        )
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative.")
    if not math.isfinite(args.threshold):
        raise ValueError("--threshold must be finite.")
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be at least 1.")
    if not math.isfinite(args.duration) or args.duration < 0:
        raise ValueError("--duration must be a finite, non-negative number of seconds.")
    if not math.isfinite(args.reconnect_interval) or args.reconnect_interval < 0:
        raise ValueError("--reconnect-interval must be finite and non-negative.")
    if args.max_reconnect_attempts < 1:
        raise ValueError("--max-reconnect-attempts must be at least 1.")
    if args.open_timeout_ms < 1 or args.read_timeout_ms < 1:
        raise ValueError("RTSP open/read timeouts must be positive milliseconds.")
    if not math.isfinite(args.recording_fps) or args.recording_fps <= 0:
        raise ValueError("--recording-fps must be a positive finite number.")
    if (args.display_width is None) != (args.display_height is None):
        raise ValueError(
            "--display-width and --display-height must be provided together."
        )
    if args.display_width is not None and (
        args.display_width < 1 or args.display_height < 1
    ):
        raise ValueError("Display dimensions must be positive.")
    if args.mode != "rtsp" and args.display_width is not None:
        raise ValueError("Display dimensions are only supported in RTSP mode.")

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"P2PNet checkpoint does not exist: {checkpoint_path}")

    device = resolve_device(args.device)

    if args.mode == "rtsp":
        if args.duration == 0 and args.max_frames == 0 and args.no_display:
            print(
                "RTSP monitoring has no duration or frame limit and no display; "
                "press Ctrl+C to stop."
            )
        if args.record and args.output is None:
            args.output = (
                MODEL_ROOT / "outputs" / "rtsp" / "rtsp_crowd_count_output.mp4"
            )
        if args.output is not None and not args.record:
            raise ValueError("--output in RTSP mode requires --record.")
        source = resolve_rtsp_source(args)
        if not ffmpeg_backend_available():
            raise RuntimeError(
                "RTSP mode requires OpenCV built with the FFmpeg video backend."
            )
        if args.threads:
            torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        model = load_model(checkpoint_path, device)
        if args.channels_last:
            if device.type != "cpu":
                raise ValueError("--channels-last is only supported for CPU inference.")
            model.to(memory_format=torch.channels_last)
        print_runtime_summary(checkpoint_path, device)
        print_device_summary(device)
        run_rtsp(args, model, source)
        return

    if args.source:
        raise ValueError("--source is only used with --mode rtsp.")
    if args.record:
        raise ValueError("--record is only used with --mode rtsp.")
    if args.duration > 0:
        raise ValueError("--duration is only used with --mode rtsp.")
    if args.output is None:
        args.output = Path("outputs/park_video/park_crowd_count_output.mp4")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must use the .mp4 extension.")

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    csv_path = output_path.parent / "park_frame_counts.csv"
    summary_path = output_path.parent / "park_video_summary.json"
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("The output video path must differ from the input path.")
    final_paths = (output_path, csv_path, summary_path)
    existing = [path for path in final_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing video results: "
            + ", ".join(str(path) for path in existing)
            + ". Use --overwrite to replace them."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_tag = f"{os.getpid()}.{time.time_ns()}.partial"
    temporary_video = output_path.with_name(
        f"{output_path.stem}.{temporary_tag}{output_path.suffix}"
    )
    temporary_csv = csv_path.with_name(
        f"{csv_path.stem}.{temporary_tag}{csv_path.suffix}"
    )
    temporary_summary = summary_path.with_name(
        f"{summary_path.stem}.{temporary_tag}{summary_path.suffix}"
    )

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = load_model(checkpoint_path, device)
    if args.channels_last:
        if device.type != "cpu":
            raise ValueError("--channels-last is only supported for CPU inference.")
        model.to(memory_format=torch.channels_last)
    print_runtime_summary(checkpoint_path, device)
    print_device_summary(device)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"OpenCV could not open input video '{input_path}'. "
            "Check that the file is a supported video format and its codec is available."
        )

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError(
            f"Input video reports an invalid frame rate ({source_fps})."
        )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 1 or height < 1:
        capture.release()
        raise RuntimeError(
            f"Input video reports invalid dimensions ({width}x{height})."
        )

    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(
            f"OpenCV could not create MP4 output '{output_path}'. "
            "Try a different output directory or codec-capable OpenCV build."
        )

    count_stats = RunningStats()
    inference_stats = RunningStats()
    preprocessing_stats = RunningStats()
    postprocessing_stats = RunningStats()
    processing_stats = RunningStats()
    cpu_stats = RunningStats()
    rolling_performance = RollingPerformance()
    resource_monitor = ProcessResourceMonitor()
    latest_resources = {
        "cpu_percent": 0.0,
        "memory_mb": 0.0,
        "peak_memory_mb": 0.0,
    }
    processed_frames = 0
    decoded_frames = 0
    video_started = time.perf_counter()
    completed = False
    try:
        with temporary_csv.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "frame_number",
                    "timestamp",
                    "estimated_crowd_count",
                    "predicted_head_point_count",
                    "preprocessing_time_ms",
                    "inference_time_ms",
                    "postprocessing_time_ms",
                    "processing_time_ms",
                    "inference_fps",
                    "processing_fps",
                    "cpu_percent",
                    "memory_mb",
                ],
            )
            csv_writer.writeheader()
            with torch.inference_mode():
                while not args.max_frames or processed_frames < args.max_frames:
                    frame_started = time.perf_counter()
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame.shape[:2] != (height, width):
                        raise RuntimeError(
                            "Input video changed frame dimensions during decoding; "
                            "a fixed-resolution MP4 cannot preserve those frames."
                        )

                    frame_index = decoded_frames
                    frame_number = frame_index + 1
                    decoded_frames += 1
                    if frame_index % args.frame_skip:
                        writer.write(frame)
                        continue

                    timestamp = timestamp_for_frame(frame_index, source_fps)
                    (
                        points,
                        inference_ms,
                        preprocessing_ms,
                        postprocessing_ms,
                    ) = prepare_and_infer(
                        frame,
                        model,
                        args.threshold,
                        args.max_long_edge,
                    )
                    count = len(points)
                    status = status_for_count(
                        count,
                        args.low_max_count,
                        args.moderate_max_count,
                    )
                    draw_overlay(
                        frame,
                        points,
                        frame_number,
                        timestamp,
                        status,
                    )
                    writer.write(frame)

                    processing_ms = (time.perf_counter() - frame_started) * 1000.0
                    count_stats.add(count)
                    inference_stats.add(inference_ms)
                    preprocessing_stats.add(preprocessing_ms)
                    postprocessing_stats.add(postprocessing_ms)
                    processing_stats.add(processing_ms)
                    processed_frames += 1
                    rolling_performance.add(
                        preprocessing_ms,
                        inference_ms,
                        postprocessing_ms,
                        processing_ms,
                    )
                    rolling_metrics = rolling_performance.snapshot()
                    latest_resources = resource_monitor.sample()
                    cpu_stats.add(latest_resources["cpu_percent"])
                    csv_writer.writerow({
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "estimated_crowd_count": count,
                        "predicted_head_point_count": len(points),
                        "preprocessing_time_ms": f"{preprocessing_ms:.3f}",
                        "inference_time_ms": f"{inference_ms:.3f}",
                        "postprocessing_time_ms": f"{postprocessing_ms:.3f}",
                        "processing_time_ms": f"{processing_ms:.3f}",
                        "inference_fps": (
                            f"{rolling_metrics['inference_fps']:.3f}"
                        ),
                        "processing_fps": (
                            f"{rolling_metrics['processing_fps']:.3f}"
                        ),
                        "cpu_percent": f"{latest_resources['cpu_percent']:.1f}",
                        "memory_mb": f"{latest_resources['memory_mb']:.1f}",
                    })
                    if frame_number == 1 or frame_number % 30 == 0:
                        print(
                            f"Frame {frame_number}: {count} people, "
                            f"inference {rolling_metrics['inference_ms']:.1f} ms "
                            f"({rolling_metrics['inference_fps']:.2f} FPS), "
                            f"pre/post {preprocessing_ms:.1f}/"
                            f"{postprocessing_ms:.1f} ms, "
                            f"processing {rolling_metrics['processing_fps']:.2f} FPS, "
                            f"CPU {latest_resources['cpu_percent']:.0f}%, "
                            f"RAM {latest_resources['memory_mb']:.0f} MiB"
                        )

        writer.release()
        capture.release()
        if processed_frames == 0:
            raise RuntimeError(f"No video frames could be decoded from '{input_path}'.")
        if (
            not args.max_frames
            and source_frames > 0
            and decoded_frames != source_frames
        ):
            raise RuntimeError(
                f"Decoded {decoded_frames} frames, but the video reports "
                f"{source_frames}; refusing to publish an incomplete output."
            )

        video_processing_seconds = max(time.perf_counter() - video_started, 1e-9)
        processed_duration = decoded_frames / source_fps
        source_duration = (
            source_frames / source_fps if source_frames > 0 else processed_duration
        )
        summary = {
            "input_video": str(input_path),
            "checkpoint": str(checkpoint_path),
            **device_summary(device),
            "output_video": str(output_path),
            "csv_results": str(csv_path),
            "total_processed_frames": processed_frames,
            "total_decoded_frames": decoded_frames,
            "source_frame_count": source_frames if source_frames > 0 else None,
            "average_estimated_crowd_count": count_stats.mean,
            "maximum_estimated_crowd_count": int(count_stats.maximum),
            "minimum_estimated_crowd_count": int(count_stats.minimum),
            "average_preprocessing_time_ms_per_frame": preprocessing_stats.mean,
            "average_inference_time_ms_per_frame": inference_stats.mean,
            "average_postprocessing_time_ms_per_frame": postprocessing_stats.mean,
            "average_total_processing_time_ms_per_frame": processing_stats.mean,
            "average_processing_fps": (
                processed_frames / video_processing_seconds
                if video_processing_seconds > 0
                else 0.0
            ),
            "average_decoded_frame_fps": (
                decoded_frames / video_processing_seconds
                if video_processing_seconds > 0
                else 0.0
            ),
            "video_processing_wall_time_seconds": video_processing_seconds,
            "average_inference_fps": (
                processed_frames * 1000.0 / inference_stats.total
                if inference_stats.total > 0
                else 0.0
            ),
            "rolling_window_frames": 30,
            "final_rolling_performance": rolling_performance.snapshot(),
            "average_cpu_percent": cpu_stats.mean,
            "memory_mb": latest_resources["memory_mb"],
            "peak_memory_mb": latest_resources["peak_memory_mb"],
            "video_duration_seconds": source_duration,
            "processed_output_duration_seconds": processed_duration,
            "source_fps": source_fps,
            "output_fps": source_fps,
            "frame_width": width,
            "frame_height": height,
            "confidence_threshold": args.threshold,
            "frame_skip": args.frame_skip,
            "channels_last": args.channels_last,
            "crowd_status_thresholds": {
                "low": f"0-{args.low_max_count}",
                "moderate": f"{args.low_max_count + 1}-{args.moderate_max_count}",
                "high": f"above {args.moderate_max_count}",
            },
            "crowd_status_note": (
                "Illustrative demonstration thresholds only; not validated "
                "safety thresholds."
            ),
            "timing_note": (
                "Inference time measures the P2PNet forward pass only. Total "
                "processing time includes frame decoding, preprocessing, model "
                "inference, postprocessing, overlay drawing, and video writing; "
                "it excludes model loading and CSV/summary bookkeeping."
            ),
            "frame_limit_applied": args.max_frames or None,
        }
        with temporary_summary.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)

        temporary_video.replace(output_path)
        temporary_csv.replace(csv_path)
        temporary_summary.replace(summary_path)
        completed = True
        print(json.dumps(summary, indent=2))
        print(f"Annotated video: {output_path}")
        print(f"Frame counts CSV: {csv_path}")
        print(f"Summary: {summary_path}")
    finally:
        capture.release()
        writer.release()
        if not completed:
            for path in (temporary_video, temporary_csv, temporary_summary):
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    main()
