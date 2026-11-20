# JHU-CROWD++ v2.0 integration

The [official JHU-CROWD++ download page](https://www.crowd-counting.com/#download)
links to the [public Google Drive folder](https://drive.google.com/drive/folders/1FkdvHyAom1B2aVj6_jZpZPW01sQNiI7n).
The archive was extracted here without modifying source files. The original
archive is retained at `P2PNet/dataset/_incoming/jhu_crowd_v2.0.zip`; its size is
3,080,825,298 bytes and SHA-256 is
`886FBB75AB543977E12FE8C0491FF52E728DA37B93457A03F8878682E70075E9`.
The original `README` and `License` are preserved with the extracted data. Google
Drive indicated it could not virus-scan this 2.9 GB archive; its ZIP CRCs passed
and its member paths were checked before extraction. No archive contents were run.

## Dataset layout and annotations

```text
JHU-CROWD++/
  train/images/  2,272 images
  train/gt/      2,272 point-label files
  train/image_labels.txt
  val/images/      500 images
  val/gt/          500 point-label files
  val/image_labels.txt
  test/images/   1,600 images
  test/gt/       1,600 point-label files
  test/image_labels.txt
  annotations/counts.csv          generated count manifest
  shanghai_tech_part_a_train.list generated P2PNet train manifest
  shanghai_tech_part_a_test.list  generated P2PNet validation manifest
  jhu_crowd_test.list             generated P2PNet test manifest
  validation_report.json          generated validation results
```

Each original `gt/*.txt` row is `x y w h occlusion blur`. The first two values
are the annotated head point used by point-counting models. The other fields are
preserved metadata; this integration does not create or train on bounding boxes.
`annotations/counts.csv` is derived from these point rows and cross-checked against
the official per-image counts in each split's `image_labels.txt`.

The original data remains unchanged. The generated list files reference the
original image and annotation files. P2PNet's SHHA loader reads the first two
values from each point row. Its path-to-image-ID handling was made Windows-safe.

## Validation status

Run the preparation and validation script from `P2PNet`:

```powershell
.\.venv-py313\Scripts\python.exe .\preprocessing\prepare_jhu_crowd.py
```

The archive passed ZIP CRC verification. The dataset contains 4,372 images,
4,372 point files, and 1,515,005 point rows. All image counts match the point-row
counts. The validator also found **490 head coordinates outside image bounds**;
they were not corrected or removed. It reports these and returns a non-zero exit
status while still generating count/list manifests for evaluation and inspection.
Do not train on the point files until these coordinates have been reviewed.

The head-width/height metadata contains 14,964 rows with a non-positive value,
including two with a negative value. Those box dimensions are not used by this
point-counting integration; the original rows are preserved.

## License and use

The included license permits academic and non-commercial use only. It prohibits
commercial use of the images and derived data. Do not use this dataset or
predictions derived from it for commercial CCTV deployment, licensing, or
commercial gain. Cite the papers listed in the included `README`.

## Inference

All four existing inference wrappers can read images recursively from this shared
dataset root. Running inference over all 4,372 images on the local CPU is expected
to take many hours and was not started. Use separate output paths to preserve the
existing 16-frame results. For example, from `P2PNet`:

```powershell
.\.venv-py313\Scripts\python.exe .\inference\run_frames.py `
  --frames-dir .\dataset\JHU-CROWD++ `
  --predictions .\outputs\predictions\jhu_crowd_predictions.csv `
  --visualizations-dir .\outputs\evaluation\jhu_crowd\annotated_predictions

.\.venv-py313\Scripts\python.exe .\evaluation\evaluate_counts.py `
  --dataset-root .\dataset\JHU-CROWD++ `
  --predictions .\outputs\predictions\jhu_crowd_predictions.csv `
  --output-dir .\outputs\evaluation\jhu_crowd `
  --checkpoint .\original_repo\weights\SHTechA.pth `
  --pretrained-weights-used
```

The existing model comparison and per-model results still refer to the original
16-frame sample; no JHU-CROWD++ inference or model evaluation has been run.
