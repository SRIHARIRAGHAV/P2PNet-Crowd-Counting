import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))

from predict_video import (
    LatestFrameReader,
    automatic_display_size,
    draw_head_points,
    fit_frame_to_display,
    load_model,
    map_points_to_display,
    parse_args,
    resolve_device,
    resolve_rtsp_source,
    rtsp_output_paths,
    valid_prediction_points,
)
from models.p2pnet import AnchorPoints


class FakeCapture:
    def __init__(self, frames=(), opened=True, delay=0.0):
        self.frames = list(frames)
        self.opened = opened
        self.delay = delay
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        return 15.0

    def set(self, property_id, value):
        return True

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


class RtspSupportTests(unittest.TestCase):
    def test_device_auto_uses_cuda_only_when_available(self):
        with mock.patch("predict_video.torch.cuda.is_available", return_value=False):
            self.assertEqual(str(resolve_device("auto")), "cpu")

        with mock.patch(
            "predict_video.torch.cuda.is_available", return_value=True
        ), mock.patch(
            "predict_video.torch.cuda.current_device", return_value=0
        ), mock.patch(
            "predict_video.torch.cuda.device_count", return_value=1
        ):
            self.assertEqual(str(resolve_device("auto")), "cuda:0")
            self.assertEqual(str(resolve_device("cuda")), "cuda:0")
            self.assertEqual(str(resolve_device("cuda:0")), "cuda:0")

    def test_explicit_cuda_device_fails_when_unavailable(self):
        with mock.patch("predict_video.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                resolve_device("cuda")

    def test_checkpoint_is_loaded_on_cpu_before_model_device_transfer(self):
        model = torch.nn.Linear(2, 2)
        with mock.patch(
            "predict_video.build_model",
            return_value=model,
        ), mock.patch(
            "predict_video.torch.load",
            return_value=model.state_dict(),
        ) as load_checkpoint:
            loaded_model = load_model(Path("checkpoint.pth"), torch.device("cpu"))

        load_checkpoint.assert_called_once_with(
            "checkpoint.pth",
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(next(loaded_model.parameters()).device.type, "cpu")

    def test_anchor_points_follow_input_device_even_when_cuda_is_available(self):
        image = torch.zeros(1, 3, 128, 128)
        anchor_points = AnchorPoints(pyramid_levels=[3], row=2, line=2)
        with mock.patch("models.p2pnet.torch.cuda.is_available", return_value=True):
            points = anchor_points(image)
        self.assertEqual(points.device, image.device)

    def test_rtsp_url_can_be_read_from_environment(self):
        variable = "P2PNET_TEST_RTSP_URL"
        previous = os.environ.get(variable)
        os.environ[variable] = "rtsp://camera-user:secret@camera.local:554/live"
        try:
            source = resolve_rtsp_source(
                type("Args", (), {"source": None, "source_env": variable})()
            )
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
        self.assertIn("secret@", source)

    def test_invalid_rtsp_urls_are_rejected(self):
        for source in (
            "http://camera.local/live",
            "rtsp:///live",
            "rtsp://camera.local:70000/live",
            "rtsp://camera.local:not-a-port/live",
        ):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    resolve_rtsp_source(
                        type("Args", (), {"source": source, "source_env": ""})()
                    )

    def test_existing_rtsp_results_get_unique_paths_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "rtsp"
            output_dir.mkdir()
            csv_path = output_dir / "rtsp_frame_counts.csv"
            summary_path = output_dir / "rtsp_video_summary.json"
            video_path = output_dir / "rtsp_crowd_count_output.mp4"
            csv_path.write_text("existing csv", encoding="utf-8")
            summary_path.write_text("existing summary", encoding="utf-8")
            video_path.write_bytes(b"existing video")

            first_paths = rtsp_output_paths(output_dir, video_path, overwrite=False)
            self.assertTrue(all(path is not None for path in first_paths))
            self.assertNotEqual(first_paths, (csv_path, summary_path, video_path))
            self.assertEqual(csv_path.read_text(encoding="utf-8"), "existing csv")
            self.assertEqual(
                summary_path.read_text(encoding="utf-8"),
                "existing summary",
            )
            self.assertEqual(video_path.read_bytes(), b"existing video")

            first_paths[0].touch()
            first_paths[1].touch()
            first_paths[2].touch()
            second_paths = rtsp_output_paths(output_dir, video_path, overwrite=False)
            self.assertNotEqual(first_paths, second_paths)

    def test_display_fit_preserves_full_frame_and_point_mapping(self):
        cases = (
            (1920, 1080, (1280, 720)),
            (1280, 720, (1000, 1000)),
            (2560, 1440, (800, 600)),
            (3840, 2160, (1280, 720)),
            (720, 1280, (1280, 720)),
        )
        for width, height, display_size in cases:
            with self.subTest(source=(width, height), display=display_size):
                frame = np.full((height, width, 3), 180, dtype=np.uint8)
                display, transform = fit_frame_to_display(frame, display_size)
                (
                    scale_x,
                    scale_y,
                    offset_x,
                    offset_y,
                    resized_width,
                    resized_height,
                ) = transform
                self.assertEqual(
                    display.shape[:2],
                    (display_size[1], display_size[0]),
                )
                self.assertEqual(
                    tuple(display[offset_y, offset_x]),
                    (180, 180, 180),
                )
                self.assertEqual(
                    tuple(
                        display[
                            offset_y + resized_height - 1,
                            offset_x + resized_width - 1,
                        ]
                    ),
                    (180, 180, 180),
                )
                self.assertAlmostEqual(
                    resized_width / resized_height,
                    width / height,
                    delta=0.005,
                )

                source_points = np.array(
                    [[0, 0], [width - 1, height - 1], [width / 2, height / 2]],
                    dtype=np.float32,
                )
                mapped_points = map_points_to_display(source_points, transform)
                self.assertEqual(len(mapped_points), len(source_points))
                self.assertTrue(np.all(mapped_points[:, 0] >= offset_x))
                self.assertTrue(
                    np.all(mapped_points[:, 0] < offset_x + resized_width)
                )
                self.assertTrue(np.all(mapped_points[:, 1] >= offset_y))
                self.assertTrue(
                    np.all(mapped_points[:, 1] < offset_y + resized_height)
                )
                marked_display = display.copy()
                draw_head_points(marked_display, mapped_points)
                for x, y in mapped_points:
                    self.assertTrue(
                        np.array_equal(
                            marked_display[round(float(y)), round(float(x))],
                            [0, 0, 255],
                        )
                    )

    def test_automatic_display_size_fits_screen_without_upscaling(self):
        with mock.patch(
            "predict_video.available_display_bounds",
            return_value=(1280, 720),
        ):
            self.assertEqual(automatic_display_size(3840, 2160), (1280, 720))
            self.assertEqual(automatic_display_size(640, 480), (640, 480))

    def test_json_config_defaults_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "inference.json"
            config_path.write_text(
                json.dumps({
                    "mode": "video",
                    "input": "dataset/video/park_video.mp4",
                    "checkpoint": "original_repo/weights/SHTechA.pth",
                    "device": "auto",
                    "max_long_edge": 768,
                }),
                encoding="utf-8",
            )
            args = parse_args([
                "--config",
                str(config_path),
                "--device",
                "cpu",
            ])

        self.assertEqual(args.input, MODEL_ROOT / "dataset/video/park_video.mp4")
        self.assertEqual(
            args.checkpoint,
            MODEL_ROOT / "original_repo/weights/SHTechA.pth",
        )
        self.assertEqual(args.max_long_edge, 768)
        self.assertEqual(args.device, "cpu")

    def test_json_config_rejects_credentials_and_unknown_keys(self):
        cases = (
            {"source": "rtsp://user:password@camera.local/live"},
            {"unexpected_option": True},
        )
        for values in cases:
            with self.subTest(values=values):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config_path = Path(temp_dir) / "inference.json"
                    config_path.write_text(json.dumps(values), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        parse_args(["--config", str(config_path)])

    def test_latest_frame_reader_bounds_queue_and_reconnects(self):
        frames = [
            np.full((8, 8, 3), value, dtype=np.uint8)
            for value in range(5)
        ]
        captures = []

        def capture_factory(source, backend, parameters):
            self.assertEqual(backend, cv2.CAP_FFMPEG)
            self.assertIn(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, parameters)
            capture = (
                FakeCapture(frames, delay=0.01)
                if not captures
                else FakeCapture(opened=False)
            )
            captures.append(capture)
            return capture

        reader = LatestFrameReader(
            "rtsp://camera.local/live",
            reconnect_interval=0.01,
            max_reconnect_attempts=1,
            open_timeout_ms=100,
            read_timeout_ms=100,
            capture_factory=capture_factory,
        )
        reader.start()
        observed_sequence = 0
        deadline = time.monotonic() + 2.0
        finished = False
        while time.monotonic() < deadline and not finished:
            (
                observed_sequence,
                _,
                _,
                _,
                finished,
                _,
            ) = reader.get_latest(observed_sequence, timeout=0.05)
        reader.stop()
        sequence, latest_frame, _, status, _, _ = reader.get_latest(0, timeout=0)

        self.assertTrue(finished)
        self.assertEqual(sequence, len(frames))
        self.assertEqual(int(latest_frame[0, 0, 0]), len(frames) - 1)
        self.assertEqual(reader.reconnection_attempts, 1)
        self.assertEqual(status, "DISCONNECTED")
        self.assertTrue(all(capture.released for capture in captures))

    def test_valid_point_filter_matches_drawable_points(self):
        outputs = {
            "pred_logits": torch.tensor([[[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]]]),
            "pred_points": torch.tensor([[[4.0, 5.0], [20.0, 2.0], [float("nan"), 1.0]]]),
        }
        points = valid_prediction_points(
            outputs,
            threshold=0.5,
            original_size=(10, 10),
            resized_size=(10, 10),
        )
        self.assertEqual(points.shape, (1, 2))
        self.assertTrue(np.allclose(points[0], [4.0, 5.0]))


if __name__ == "__main__":
    unittest.main()
