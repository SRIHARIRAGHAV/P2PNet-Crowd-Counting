# P2PNet Crowd Counting

This project packages the existing Tencent P2PNet implementation with image,
prerecorded-video, and live RTSP inference helpers. It predicts individual head
points and counts the valid points in each frame. The P2PNet architecture and
trained checkpoint are not replaced or retrained by the inference commands.

> **Use and licensing:** The upstream P2PNet license restricts use and
> redistribution to academic research. JHU-CROWD++ has separate academic and
> non-commercial restrictions. There is no project-wide license for this
> repository. Read [licensing and publishing notes](docs/LICENSING_AND_PUBLISHING.md)
> before use or publication; this is not legal advice.

## Features

- Original P2PNet VGG16-based point-prediction architecture and strict
  checkpoint loading.
- CPU inference, automatic CPU/CUDA selection, and explicit CUDA selection
  when supported by the installed PyTorch build and NVIDIA hardware.
- Image inference and count evaluation.
- MP4/AVI/MOV prerecorded-video inference with point overlays and CSV/JSON
  reports.
- RTSP/RTSPS inference through OpenCV's FFmpeg backend, latest-frame
  acquisition, bounded timeouts, finite reconnection, optional annotated MP4
  recording, and frame reports.
- Aspect-ratio-preserving live display with configurable dimensions. Recording
  retains source-frame dimensions independently of display resizing.
- JSON examples for video and RTSP options. Camera URLs and credentials are
  supplied locally through environment variables, not committed config.

The software reports measured processing speed; it does not guarantee
real-time performance or provide safety monitoring. LOW/MODERATE/HIGH count
bands are demonstration thresholds, not validated safety limits.

## Project layout

```text
P2PNet/
  configs/                 Example inference JSON
  checkpoints/             Checkpoint download and verification instructions
  dataset/                 Dataset placement notes and retained manifests
  docs/                    Licensing and publication checklist
  evaluation/              Count evaluation and reports
  inference/               Shared device selection and image inference
  original_repo/            Upstream P2PNet model, training and license
  preprocessing/            Dataset validation and list-generation scripts
  scripts/                 CPU setup scripts for Windows and Linux
  tests/                   RTSP/display/configuration unit tests
  predict_video.py          Prerecorded video and RTSP inference
  requirements*.txt         Common, CPU, and CUDA dependency setup
```

The raw datasets, local video files, model checkpoints, virtual environments,
and generated results are intentionally excluded from Git by default. The
provided park video and image examples are not included in a clone.

## Tested environment and requirements

The dependency pins include Python 3.10-compatible Linux wheels, so Ubuntu
Python 3.10.x can install and run CPU inference. The current pins also support
Python 3.13. Python 3.10 on Ubuntu has not been executed in this workspace;
validate checkpoint loading and inference on the target machine before
deployment. CUDA inference, full training, and access to a physical RTSP
camera are not claimed as verified. Dependencies are split so that GPU setup
does not silently replace a CUDA-enabled PyTorch build:

- `requirements.txt`: shared NumPy, SciPy, Matplotlib, Pillow, OpenCV, and
  TensorBoardX dependencies.
- `requirements-cpu.txt`: shared dependencies plus CPU-only PyTorch and
  torchvision wheels.
- `requirements-gpu.txt`: shared dependencies only. Install CUDA-compatible
  PyTorch and torchvision separately first.

### CPU setup on Windows

From the project directory in PowerShell:

```powershell
.\scripts\setup_cpu_windows.ps1
.\.venv-cpu\Scripts\python.exe -m pip check
.\.venv-cpu\Scripts\python.exe -c "import torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, cv2.__version__, torch.cuda.is_available())"
```

The last command should report a CPU PyTorch build and `False` for CUDA.

### CPU setup on Linux

Install Python 3.10 and its `venv` package first (for example,
`python3.10-venv` on Ubuntu), then run:

```bash
bash scripts/setup_cpu_linux.sh
.venv-cpu/bin/python -m pip check
.venv-cpu/bin/python -c "import torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, cv2.__version__, torch.cuda.is_available())"
```

### NVIDIA CUDA setup

Install a current NVIDIA driver, then use the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) to
choose the CUDA-compatible PyTorch and torchvision command for your OS and
Python. Create a separate environment and install those wheels before the
shared project requirements:

```powershell
py -3.13 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
# Run the torch/torchvision command produced by the official PyTorch selector.
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv-gpu\Scripts\python.exe -m pip check
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device')"
```

Do not install `requirements-cpu.txt` in this GPU environment. Use `--device
cuda` only after `torch.cuda.is_available()` reports `True`; `--device auto`
selects CUDA when it is available and CPU otherwise. An explicit CUDA request
fails clearly when CUDA cannot be used. The existing model architecture and
checkpoint format do not depend on the inference device.

## Checkpoint

Inference requires the official ShanghaiTech Part A checkpoint
`original_repo/weights/SHTechA.pth`. It is not included in Git. Obtain it from
the [official upstream location](https://raw.githubusercontent.com/TencentYoutuResearch/CrowdCounting-P2PNet/main/weights/SHTechA.pth)
only if your use complies with its license, and place it at that path. The
checkpoint guide documents the expected size, SHA-256, verification commands,
and how to select another compatible checkpoint:
[checkpoints/README.md](checkpoints/README.md).

The inference loaders use safe CPU checkpoint deserialization and strict
state-dictionary matching. Missing, corrupt, or incompatible weights are
reported as errors; inference never silently falls back to random weights.
Supply another checkpoint with `--checkpoint` to `predict_video.py` or
`--weight` to `inference/run_frames.py`.

## Image inference and count evaluation

Supply a local directory of images; each image is processed independently to
avoid loading the entire dataset into memory. For example:

```powershell
.\.venv-cpu\Scripts\python.exe .\inference\run_frames.py `
  --frames-dir "C:\data\crowd_images" `
  --weight .\original_repo\weights\SHTechA.pth `
  --device cpu `
  --threads 1
```

The script writes predicted counts, inference time, and head-point coordinates
to `outputs/predictions/crowd_point_annotations_predictions.csv`, and annotated
images to `outputs/evaluation/annotated_predictions/`. The inference resize
defaults to a 1024-pixel maximum long edge; coordinates are mapped back to the
original image dimensions.

To evaluate predictions against the project's point-count manifest:

```powershell
.\.venv-cpu\Scripts\python.exe .\evaluation\evaluate_counts.py `
  --dataset-root .\dataset\crowd_counting `
  --predictions .\outputs\predictions\crowd_point_annotations_predictions.csv `
  --checkpoint .\original_repo\weights\SHTechA.pth `
  --pretrained-weights-used
```

Evaluation requires the relevant local dataset images, annotations, and
manifests. See [dataset/README.md](dataset/README.md) for directory layouts,
dataset restrictions, and preparation scripts.

## Prerecorded video inference

`predict_video.py` accepts common MP4, AVI, or MOV inputs when the installed
OpenCV backend can decode their codecs. Provide a local video path and choose
the inference device:

```powershell
.\.venv-cpu\Scripts\python.exe .\predict_video.py `
  --input "C:\data\park_video.mp4" `
  --checkpoint .\original_repo\weights\SHTechA.pth `
  --device cpu `
  --output .\outputs\park_video\park_crowd_count_output.mp4 `
  --max-frames 2
```

Remove `--max-frames 2` to process the complete video. Omit `--device` for
automatic selection or use `--device cuda` on a verified CUDA environment.
`--threshold` controls the foreground point-confidence threshold, and
`--max-long-edge` controls inference resolution while points are mapped back
to the source frame.

The output MP4 retains source dimensions and frame rate. A sibling
`park_frame_counts.csv` records per-frame counts and timings; a sibling
`park_video_summary.json` records aggregate metrics. Inference time measures
the model forward pass; total processing time includes decoding, preprocessing,
postprocessing, drawing, and video writing. Existing output paths are
protected: select a new output path or explicitly pass `--overwrite`.

## Live RTSP inference

The camera or NVR must be reachable over the network, have RTSP enabled, and
provide an account with permission to view the stream. Obtain the URL and stream
path from the device configuration or vendor documentation. Codec support
depends on the installed OpenCV FFmpeg backend; H.265 is not guaranteed.

For safer credential handling, place the URL in an environment variable rather
than a committed file or command line. In PowerShell, set it for the current
session (replace placeholders locally; do not share the value):

```powershell
$env:P2PNET_RTSP_URL = "rtsp://<username>:<password>@<camera-host>:554/<stream-path>"
```

Start monitoring; press `q` in the display window or `Ctrl+C` to stop:

```powershell
.\.venv-cpu\Scripts\python.exe .\predict_video.py `
  --mode rtsp `
  --source-env P2PNET_RTSP_URL `
  --device cpu
```

For optional annotated recording or a five-minute bounded run:

```powershell
.\.venv-cpu\Scripts\python.exe .\predict_video.py `
  --mode rtsp `
  --source-env P2PNET_RTSP_URL `
  --device cpu `
  --record `
  --duration 300
```

The default RTSP outputs are:

- `outputs/rtsp/rtsp_crowd_count_output.mp4` when `--record` is enabled.
- `outputs/rtsp/rtsp_frame_counts.csv`.
- `outputs/rtsp/rtsp_video_summary.json`.

Recording preserves the incoming frame resolution; display resizing does not
change the saved video. The CSV separates inference time from total processing
time and records count, points, FPS, timestamp, and camera status. The JSON
summary includes processed frames, count statistics, timing averages, total
monitoring duration, and reconnect attempts. It does not contain the source
URL. Existing results are preserved; when default results already exist, a
timestamped set of output names is selected automatically.

The reader uses OpenCV's FFmpeg backend, bounded open/read timeouts, TCP
transport, a one-frame capture buffer where available, and a latest-frame
strategy. It retries temporary failures up to `--max-reconnect-attempts`.
Useful options:

| Option | Default | Description |
|---|---:|---|
| `--duration` | `0` | Monitoring duration in seconds; zero means until stopped |
| `--frame-skip` | `1` | Process every Nth received frame |
| `--max-long-edge` | `1024` | Maximum inference image edge, not recording size |
| `--reconnect-interval` | `3` | Delay between connection attempts, seconds |
| `--max-reconnect-attempts` | `10` | Finite reconnect limit |
| `--open-timeout-ms` / `--read-timeout-ms` | `5000` / `3000` | FFmpeg connection/read timeouts |
| `--recording-fps` | `15` | Fallback when stream FPS is unavailable |
| `--display-width` / `--display-height` | automatic | Initial display bounds; set both |

The live window fits the full camera frame into its viewport with preserved
aspect ratio and letterboxing. Head points are mapped from source-frame
coordinates to display coordinates; display resizing does not affect counts.
Recording and display dimensions are independent.

Configuration examples are in `configs/`. Use `--config
configs/rtsp.example.json` for non-secret settings; config paths are relative
to the project root and explicit command-line values take precedence. RTSP
URLs are rejected in JSON config files. Direct `--source "rtsp://..."` is
supported but may expose credentials in shell history or process listings.

### RTSP troubleshooting

- Verify the URL scheme, device address, port, stream path, account
  permissions, RTSP setting, firewall, and VPN.
- Ensure OpenCV reports FFmpeg support. If FFmpeg is unavailable or cannot
  decode the codec, install a compatible OpenCV build or use a camera stream
  codec supported by that FFmpeg build.
- Check that the output folder is writable and the OpenCV build can write
  `mp4v` MP4 files if recording fails.
- Use `--no-display` on a headless computer; set `--duration` or stop with
  `Ctrl+C`.
- Do not paste credential-bearing RTSP URLs into issue reports or logs.

## Dataset setup and training

Datasets are not downloaded or bundled automatically. Follow
[dataset/README.md](dataset/README.md) for the expected P2PNet list/annotation
format and the JHU-CROWD++ license and layout. The raw JHU dataset, local
images/videos, and point annotations are excluded from Git.

The upstream `original_repo/train.py` is CUDA-oriented; CPU training and a
complete training workflow are not verified or claimed by this publication
prep. Training requires a prepared dataset and compatible CUDA PyTorch
installation. When the upstream training path requests pretrained VGG
initialization, it fetches hash-checked weights from the official PyTorch model
host. The inference architecture itself is unchanged. Do not treat the small
local sample or an unreviewed dataset as evidence of generalization.

## Tests and verification

Run the focused automated suite from the project root:

```powershell
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover device selection behavior, checkpoint-independent point
filtering, URL/config validation, simulated latest-frame and reconnect
behavior, output-name collision handling, and display aspect-ratio/point
mapping. They do not prove access to a real camera or CUDA availability.

For a one/two-frame end-to-end video smoke test, use a local test video and
checkpoint:

```powershell
.\.venv-cpu\Scripts\python.exe .\predict_video.py `
  --input "C:\data\sample.mp4" `
  --checkpoint .\original_repo\weights\SHTechA.pth `
  --device cpu `
  --max-frames 2 `
  --output .\outputs\smoke\smoke.mp4
```

Do not claim CUDA or physical RTSP validation unless those environments have
actually been tested.

## References and publication

- X. Tian et al., “Rethinking Counting and Localization in Crowds: A Purely
  Point-Based Framework,” ICCV 2021,
  [paper](https://arxiv.org/abs/2107.12746).
- [Official Tencent P2PNet repository](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)
  and its included [`original_repo/LICENSE`](original_repo/LICENSE).
- JHU-CROWD++ data is subject to the license provided by its authors; see the
  local `dataset/JHU-CROWD++/License` when the dataset is obtained.

There is deliberately no root `LICENSE`: the project combines upstream
research-restricted code, separately restricted datasets, and project
additions without a single established license. Review
[docs/LICENSING_AND_PUBLISHING.md](docs/LICENSING_AND_PUBLISHING.md) and the
staged file list before publishing. No repository was initialized and no
remote was pushed as part of this preparation.
