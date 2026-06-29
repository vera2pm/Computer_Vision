# Tracking

Person tracking on the MOT20 dataset using YOLO (detector) + FiftyOne (dataset management & evaluation).

## Data split

| Split | Sequences | GT | Purpose |
|---|---|---|---|
| train | MOT20-01, MOT20-02 | ✓ | YOLO training |
| val | MOT20-03 | ✓ | Monitored during training; use for hyperparameter tuning |
| labeled_test | MOT20-05 | ✓ | **Held out** — run evaluation here only once, after all tuning |
| test | MOT20-04, 06, 07, 08 | ✗ | Inference / visual inspection only |

## Code structure

```
tracking/
  tracking.ipynb          ← main notebook (run everything from here)
  src/
    tracking_51.py        ← Dataset class, tracking, evaluation (FiftyOne-based)
    tracking.py           ← train_detector()
```

---

## Local usage

### Prerequisites

```bash
pip install -r requirements.txt
```

Download MOT20 from [motchallenge.net](https://motchallenge.net/data/MOT20/) and place it at:

```
data/
  MOT20/
    train/  MOT20-01/  MOT20-02/  MOT20-03/  MOT20-05/
    test/   MOT20-04/  MOT20-06/  MOT20-07/  MOT20-08/
```

### Run

Open `tracking/tracking.ipynb` and set:

```python
ENVIRONMENT = "local"
DEVICE = "mps"   # or "cuda" / "cpu"
```

Then run cells top to bottom.

---

## Google Colab usage

### Step 1 — Upload data to Google Drive

Place the MOT20 folder on your Drive so the path is:

```
MyDrive/ComputerVision/data/MOT20/
```

### Step 2 — Upload the code

Either clone the repo to Drive or upload the `tracking/` folder manually:

```
MyDrive/ComputerVision/tracking/src/tracking_51.py
MyDrive/ComputerVision/tracking/src/tracking.py
MyDrive/ComputerVision/tracking/tracking.ipynb
```

### Step 3 — Open the notebook in Colab

Upload `tracking/tracking.ipynb` to Colab (File → Upload notebook), or open it directly from Drive.

### Step 4 — Configure paths

In the **Environment setup** cell, set:

```python
ENVIRONMENT = "colab"
DATA_ROOT   = "/content/drive/MyDrive/ComputerVision/data/"
REPO_ROOT   = "/content/drive/MyDrive/ComputerVision/"
RUNS_DIR    = "/content/drive/MyDrive/ComputerVision/tracking/runs/detect"
DEVICE      = "cuda"
```

> `RUNS_DIR` on Drive means your model weights survive Colab session resets.

### Step 5 — Run cells top to bottom

The notebook installs nothing — make sure your Colab runtime has the required packages. If not, add an install cell at the top:

```python
!pip install ultralytics fiftyone sahi boxmot motmetrics
```

---

## Pipeline steps explained

| Step | What happens | Function |
|---|---|---|
| 1 | Convert MOT20 `img1/` frames → mp4 videos | `Dataset.create_videos()` |
| 2 | Load videos into persistent FiftyOne dataset `"mot20"` | `Dataset.create_dataset()` |
| 3 | Parse `gt/gt.txt` → FiftyOne frame annotations (person only, flag=1) | `Dataset.add_all_ground_truth()` |
| 4 | Export YOLO labels + copy images for train / val / labeled_test | `Dataset.save_all_labels_and_images()` |
| 5 | Copy images for no-GT test sequences | `Dataset.prepare_test_images()` |
| 6 | Write `dataset.yaml` (train + val paths only) | `Dataset.write_dataset_yaml()` |
| 7 | Train YOLO detector | `train_detector()` |
| 8 | Run tracking on val, compute MOTA/MOTP | `tracking_yolo()` + `calc_metrics()` |
| 9 | Run tracking on labeled_test, compute final metrics | `tracking_yolo()` + `calc_metrics()` |

## Changing paths

All paths are derived from a single `DATA_ROOT` variable. To change it, call `configure()` once before using any other function:

```python
from tracking_51 import configure
configure("/your/custom/data/root/")
```
