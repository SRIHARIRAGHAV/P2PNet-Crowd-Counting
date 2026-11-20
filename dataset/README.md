# Dataset setup

Raw images, videos, point annotations, and the 3 GB JHU-CROWD++ archive are not
included in Git. `.gitignore` preserves the JHU count/list manifests and dataset
license documentation while excluding local data files.

## Image inference sample layout

`inference/run_frames.py` accepts a directory and recursively processes common
image formats. Supply your own images with `--frames-dir`. The existing
`dataset/crowd_counting` manifests describe the local 16-image sample but the
images and point labels are intentionally excluded from publication.

## P2PNet training data

The upstream `SHHA` loader expects the following under a caller-supplied
`--data_root`:

```text
DATA_ROOT/
  train/
    <scene>/<image files>
    <scene>/<point annotation .txt files>
  test/
    <scene>/<image files>
    <scene>/<point annotation .txt files>
  shanghai_tech_part_a_train.list
  shanghai_tech_part_a_test.list
```

Each list row contains a relative image path followed by its annotation path.
Each annotation row is one head point as `x y` in pixel coordinates. See
`original_repo/README.md` for the upstream format and split requirements.

## JHU-CROWD++ optional dataset

The JHU-CROWD++ data was retained locally for validation only and is excluded
from Git. Obtain it from the official [JHU-CROWD++ download page](https://www.crowd-counting.com/#download)
and follow its included license, which permits academic/non-commercial use
only. Extract the `train`, `val`, and `test` directories under
`dataset/JHU-CROWD++`, then run:

```powershell
python .\preprocessing\prepare_jhu_crowd.py
```

The preparation script generates count and list manifests. The preserved local
validation report records 490 head points outside image bounds; review these
before training. Do not treat these annotations as fully validated ground
truth.

## Park video and camera data

Place a video you are authorized to process at any local path and pass it with
`predict_video.py --input`. Local videos under `dataset/video/` are ignored by
Git. RTSP URLs and credentials must remain in local environment variables, not
in committed config files.
