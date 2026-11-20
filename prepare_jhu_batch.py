import csv
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
root = MODEL_ROOT / "dataset" / "JHU-CROWD++"
base = MODEL_ROOT / "outputs" / "jhu_test_evaluation"

prediction_file = base / "predictions/batch_10.csv"
subset_root = base / "evaluation_dataset"
subset_csv = subset_root / "annotations/counts.csv"
normalized_predictions = base / "predictions/batch_10_normalized.csv"

# Read the 10 prediction IDs
with prediction_file.open("r", newline="", encoding="utf-8-sig") as f:
    predictions = list(csv.DictReader(f))

selected_ids = {
    Path(row["Frame_ID"]).stem
    for row in predictions
}

# Read original dataset counts and select matching test images only
with (root / "annotations/counts.csv").open(
    "r", newline="", encoding="utf-8-sig"
) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    selected_rows = [
        row for row in reader
        if row["split"] == "test"
        and Path(row["frame"]).stem in selected_ids
    ]

if len(selected_rows) != len(selected_ids):
    raise RuntimeError(
        f"Expected {len(selected_ids)} test rows, found {len(selected_rows)}"
    )

# Create a separate subset dataset
subset_csv.parent.mkdir(parents=True, exist_ok=True)

with subset_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(selected_rows)

# Normalize prediction IDs to match the dataset frame paths
for row in predictions:
    image_id = Path(row["Frame_ID"]).stem
    row["Frame_ID"] = f"test/images/{image_id}"

with normalized_predictions.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
    writer.writeheader()
    writer.writerows(predictions)

print("Subset dataset created:", subset_csv)
print("Normalized predictions created:", normalized_predictions)
print("Ground-truth rows:", len(selected_rows))
print("Prediction rows:", len(predictions))

for row in selected_rows:
    print(row["frame"], "Ground truth:", row["count"])
