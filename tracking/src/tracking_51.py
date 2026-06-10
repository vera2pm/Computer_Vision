import os
import shutil

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import subprocess

import fiftyone as fo
from boxmot import Boxmot
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from ultralytics.utils.ops import ltwh2xywh, xyxy2ltwh

print(f"Your FO version is: {fo.__version__}")

# ── Path configuration ──────────────────────────────────────────────────────
# All paths are derived from DATA_ROOT. Override by calling configure() once
# at the start of your script or notebook before doing anything else.
#
#   Local:  configure("../data/")                        (default, from tracking/src/)
#   Colab:  configure("/content/drive/MyDrive/ComputerVision/data/")
#
DATA_ROOT    = "../data/"
mot20_folder = os.path.join(DATA_ROOT, "MOT20")
mv_dir       = os.path.join(DATA_ROOT, "mot20_mvs_new")
out_dir      = os.path.join(DATA_ROOT, "train_mot20")
model_path   = "models/yolo11n.pt"


def configure(data_root: str = "../data/", yolo_model: str = "models/yolo11n.pt") -> None:
    """Set the data root and base YOLO model. Call this once before anything else.

    Args:
        data_root:  root folder that contains mot20_mvs/ (videos), train_mot20/ (output),
                and optionally MOT20/ (original frames). If MOT20/ is absent, frames are
                extracted from videos. gt.txt files are still required in MOT20/train/<seq>/gt/.
        yolo_model: path to base YOLO weights (downloaded automatically if missing)
    """
    global DATA_ROOT, mot20_folder, mv_dir, out_dir, model_path
    DATA_ROOT = data_root
    mot20_folder = os.path.join(data_root, "MOT20")
    mv_dir = os.path.join(data_root, "mot20_mvs")
    out_dir = os.path.join(data_root, "train_mot20")
    model_path = yolo_model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    YOLO(model_path)  # auto-downloads weights if not present
    print(f"Configured — data_root: {data_root} | model: {yolo_model}")


# ── Dataset splits ──────────────────────────────────────────────────────────
TRAIN_SEQS        = ["MOT20-01", "MOT20-02"]
VAL_SEQS          = ["MOT20-03"]
LABELED_TEST_SEQS = ["MOT20-05"]  # GT available — hold out for final evaluation only
TEST_SEQS         = ["MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"]  # no GT

GT_COLUMNS       = ["frame", "id", "x0", "y0", "w", "h", "flag", "class", "visibility"]
GT_CLASSES       = ["na", "person", "na", "na", "na", "na", "vehicle", "person",
                    "na", "na", "na", "occluded", "na", "crowd"]
PERSON_CLASS_IDS = {1, 2}


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_sequence_view(dataset, seq_name: str):
    """Return a FiftyOne view containing only the video for seq_name (e.g. 'MOT20-03')."""
    return dataset.match(fo.ViewField("filepath").contains_str(f"movi_{seq_name}"))


# ── Data preparation ─────────────────────────────────────────────────────────

class Dataset:

    @staticmethod
    def _find_img_dir(seq_path: str) -> str:
        for candidate in ("img1", "images"):
            d = os.path.join(seq_path, candidate)
            print(d)
            if os.path.isdir(d):
                return d
        raise FileNotFoundError(f"No image directory found in {seq_path}")

    @staticmethod
    def _find_sample(dataset, seq_name: str):
        for s in dataset:
            if f"movi_{seq_name}" in s.filepath:
                return s
        return None

    def create_videos(self):
        """Convert all MOT20 img1/ sequences to mp4 for FiftyOne."""
        print("Starting creating videos from images")
        os.makedirs(mv_dir, exist_ok=True)
        seqs = (
            [(s, "train") for s in TRAIN_SEQS + VAL_SEQS + LABELED_TEST_SEQS]
            + [(s, "test") for s in TEST_SEQS]
        )
        for seq_name, split in seqs:
            mov_out = os.path.join(mv_dir, f"movi_{seq_name}.mp4")
            if os.path.exists(mov_out):
                print(f"  {seq_name}: already exists, skipping")
                continue
            seq_path = os.path.join(mot20_folder, split, seq_name)
            img_dir  = self._find_img_dir(seq_path)
            print(f"  {seq_name}: creating video...")
            subprocess.run(
                ["ffmpeg", "-i", os.path.join(img_dir, "%06d.jpg"), "-c:v", "libx264", mov_out],
                check=True,
            )
            print(f"  {seq_name}: done")

    def create_dataset(self):
        """Load all videos from mv_dir into a persistent FiftyOne dataset named 'mot20'."""
        try:
            fo.load_dataset("mot20").delete()
        except ValueError:
            pass
        dataset = fo.Dataset.from_dir(dataset_dir=mv_dir, dataset_type=fo.types.VideoDirectory)
        dataset.ensure_frames()
        dataset.name = "mot20"
        dataset.persistent = True
        print(f"Dataset created with {len(dataset)} samples")
        return dataset

    def add_ground_truth(self, sample, split: str, seq_name: str):
        """Add GT person annotations from gt.txt to one FiftyOne video sample."""
        gt_file = os.path.join(mot20_folder, split, seq_name, "gt", "gt.txt")
        df = pd.read_table(gt_file, header=None, sep=",")
        df.columns = GT_COLUMNS
        df = df[(df["flag"] == 1) & (df["class"].isin(PERSON_CLASS_IDS))]
        df = df.sort_values(by="frame")

        imw    = sample.metadata.frame_width
        imh    = sample.metadata.frame_height
        frames = df["frame"].astype("int").unique()

        for frame_num in tqdm(frames, desc=f"GT {seq_name}"):
            f = sample.frames[int(frame_num)]
            f["gt"] = fo.Detections()
            for _, row in df.loc[df.frame == frame_num].iterrows():
                ltwh  = np.array([row["x0"], row["y0"], row["w"], row["h"]])
                ltwhn = np.array([ltwh[0] / imw, ltwh[1] / imh, ltwh[2] / imw, ltwh[3] / imh])
                f.gt.detections.append(
                    fo.Detection(label="person", bounding_box=ltwhn, index=int(row["id"]))
                )
            f.save()
        sample.save()

    def add_all_ground_truth(self, dataset):
        """Add GT annotations for all train, val, and labeled_test sequences."""
        for seq_name in TRAIN_SEQS + VAL_SEQS + LABELED_TEST_SEQS:
            sample = self._find_sample(dataset, seq_name)
            if sample is None:
                print(f"  {seq_name}: not found in dataset, skipping")
                continue
            print(f"  Adding GT for {seq_name}...")
            self.add_ground_truth(sample, "train", seq_name)

    def save_labels_and_images(self, sample, split: str, seq_name: str):
        """Write YOLO label files and extract images from video for one sequence."""
        seq_out = os.path.join(out_dir, split, f"movi_{seq_name}")
        lbl_dir = os.path.join(seq_out, "labels")
        img_out = os.path.join(seq_out, "images")
        os.makedirs(lbl_dir, exist_ok=True)
        os.makedirs(img_out, exist_ok=True)

        # Extract frames from video if images not already present
        video_path = os.path.join(mv_dir, f"movi_{seq_name}.mp4")
        mot20_img_src = os.path.join(mot20_folder, "train", seq_name)
        use_video = not os.path.isdir(mot20_img_src) and os.path.exists(video_path)

        if use_video:
            print(f"  {seq_name}: extracting frames from video...")
            subprocess.run(
                ["ffmpeg", "-i", video_path, os.path.join(img_out, "%06d.jpg")],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            img_src = self._find_img_dir(mot20_img_src)

        for i in tqdm(range(len(sample.frames)), desc=f"Export {seq_name}"):
            frame_number = i + 1
            frame_name   = f"{frame_number:06d}"

            if not use_video:
                src_img = os.path.join(img_src, f"{frame_name}.jpg")
                if not os.path.exists(src_img):
                    continue
                shutil.copy2(src_img, os.path.join(img_out, f"{frame_name}.jpg"))

            f = sample.frames[frame_number]
            try:
                detections = f.gt.detections if f.gt is not None else []
            except AttributeError:
                detections = []
            if not detections:
                continue

            with open(os.path.join(lbl_dir, f"{frame_name}.txt"), "w") as lf:
                for det in detections:
                    xc, yc, w, h = list(ltwh2xywh(np.array(det.bounding_box)))
                    lf.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        print(f"  {seq_name}: saved to {seq_out}")

    def save_all_labels_and_images(self, dataset):
        """Export YOLO labels + images for train, val, and labeled_test sequences."""
        for split, seqs in [("train", TRAIN_SEQS), ("val", VAL_SEQS), ("labeled_test", LABELED_TEST_SEQS)]:
            for seq_name in seqs:
                sample = self._find_sample(dataset, seq_name)
                if sample is None:
                    print(f"  {seq_name}: not found in dataset, skipping")
                    continue
                self.save_labels_and_images(sample, split, seq_name)

    def prepare_test_images(self):
        """Extract/copy images for no-GT test sequences (inference only)."""
        for seq_name in TEST_SEQS:
            img_out = os.path.join(out_dir, "test", f"movi_{seq_name}", "images")
            os.makedirs(img_out, exist_ok=True)

            mot20_seq_path = os.path.join(mot20_folder, "test", seq_name)
            video_path     = os.path.join(mv_dir, f"movi_{seq_name}.mp4")

            if os.path.isdir(mot20_seq_path):
                img_src = self._find_img_dir(mot20_seq_path)
                for fname in tqdm(sorted(os.listdir(img_src)), desc=f"Test {seq_name}"):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        shutil.copy2(os.path.join(img_src, fname), os.path.join(img_out, fname))
            elif os.path.exists(video_path):
                print(f"  {seq_name}: extracting frames from video...")
                subprocess.run(
                    ["ffmpeg", "-i", video_path, os.path.join(img_out, "%06d.jpg")],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                print(f"  {seq_name}: no source found (no MOT20 folder or video), skipping")
                continue

            print(f"  {seq_name}: images ready at {img_out}")

    def write_dataset_yaml(self):
        """Write dataset.yaml for YOLO (train + val only; labeled_test is held out)."""
        config = {
            "path":  os.path.abspath(out_dir),
            "train": [f"train/movi_{s}/images" for s in TRAIN_SEQS],
            "val":   [f"val/movi_{s}/images"   for s in VAL_SEQS],
            "nc":    1,
            "names": ["person"],
        }
        os.makedirs(out_dir, exist_ok=True)
        yaml_path = os.path.join(out_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  dataset.yaml → {yaml_path}")

    def prepare_all(self):
        """Run the full data preparation pipeline (steps 1–6)."""
        print("=== Step 1: Create videos ===")
        self.create_videos()

        print("\n=== Step 2: Create FiftyOne dataset ===")
        dataset = self.create_dataset()

        print("\n=== Step 3: Add ground truth ===")
        self.add_all_ground_truth(dataset)

        print("\n=== Step 4: Save YOLO labels and images ===")
        self.save_all_labels_and_images(dataset)

        print("\n=== Step 5: Copy test images ===")
        self.prepare_test_images()

        print("\n=== Step 6: Write dataset.yaml ===")
        self.write_dataset_yaml()

        print(f"\nDone. YOLO dataset ready at: {os.path.abspath(out_dir)}")
        return dataset


# ── Tracking ─────────────────────────────────────────────────────────────────

def tracking_yolo(view, model, tag: str, device: str = "mps"):
    """Run YOLO tracker on a FiftyOne video view and store results under `tag`."""
    mov_path = view.first().filepath
    results  = model.track(mov_path, device=device, stream=True, imgsz=1280, conf=0.4, iou=0.6)

    for frm, res in tqdm(enumerate(results), total=len(view.first().frames)):
        f = view.first().frames[frm + 1]
        f[tag] = fo.Detections()

        if res.boxes.id is None:
            continue
        is_person = res.boxes.cls.cpu().numpy() == 0
        for bb, track_id, conf in zip(
            res.boxes.xyxyn.cpu().numpy()[is_person],
            res.boxes.id.cpu().numpy()[is_person],
            res.boxes.conf.cpu().numpy()[is_person],
        ):
            det = fo.Detection(label="person", bounding_box=xyxy2ltwh(bb), index=track_id, confidence=conf)
            f[tag].detections.append(det)
        f.save()


def tracking_with_sliced_prediction(view, device: str = "mps"):
    """Run SAHI sliced detection + BotSort tracker on a FiftyOne video view."""
    frames = view.to_frames(sample_frames=True)
    imw    = view.first().metadata.frame_width
    imh    = view.first().metadata.frame_height

    detector = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.3,
        device=device,
    )
    # tracker = BotSort(reid_weights=Path("osnet_x0_25_msmt17.pt"), device=device, half=device.startswith("cuda"))
    tracker = Boxmot(
        detector="yolov8n.pt",  # specify your detector
        tracker="botsort",  # use lowercase name here
        device=device,
        reid="osnet_x0_25_msmt17",  # optional ReID model
        half = device.startswith("cuda")
    )

    for i, frame_obj in tqdm(enumerate(frames), total=len(frames)):
        f = view.first().frames[i + 1]
        f["bot_sort"] = fo.Detections()

        results = get_sliced_prediction(
            frame_obj.filepath, detection_model=detector,
            slice_height=320, slice_width=320,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        )
        if not results.object_prediction_list:
            continue
        bbs = np.array([[*res.bbox.to_xyxy(), res.score.value, res.category.id]
                        for res in results.object_prediction_list])

        tracks = tracker.update(bbs, cv2.imread(frame_obj.filepath))
        for track in tracks:
            track_id = track[4]
            ltwh     = xyxy2ltwh(track[:4])
            ltwhn    = np.array([ltwh[0] / imw, ltwh[1] / imh, ltwh[2] / imw, ltwh[3] / imh])
            try:
                label = GT_CLASSES[int(track[6])]
            except Exception:
                label = "person"
            f.bot_sort.detections.append(
                fo.Detection(label=label, bounding_box=ltwhn, index=track_id, confidence=track[5])
            )
        f.save()


# ── Evaluation ───────────────────────────────────────────────────────────────

def calc_metrics(view, tag: str):
    """Compute MOTA / MOTP against GT annotations stored in the FiftyOne view."""
    acc = mm.MOTAccumulator(auto_id=True)
    for frm in tqdm(range(len(view.first().frames)), desc="Metrics"):
        f = view.first().frames[frm + 1]

        gt_ids, gt_bb = [], []
        for d in f.gt.detections:
            gt_ids.append(d.index)
            gt_bb.append(d.bounding_box)

        track_ids, detected_bb = [], []
        for d in f[tag].detections:
            track_ids.append(d.index)
            detected_bb.append(d.bounding_box)

        acc.update(gt_ids, track_ids, mm.distances.iou_matrix(gt_bb, detected_bb, max_iou=0.6))

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["num_frames", "mota", "motp"], name="acc")
    print(summary)
    return summary


def save_results(dataset_or_view, tag: str):
    """Render tracking results as annotated video files."""
    output_dir = os.path.join(mv_dir + "_result_" + tag)
    os.makedirs(output_dir, exist_ok=True)
    dataset_or_view.draw_labels(output_dir, label_fields=[f"frames.{tag}"])
    print(f"Results saved to {output_dir}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1 — prepare data (run once)
    configure("/Users/verakocetkova/Desktop/Data Science/ComputerVision/data/")
    Dataset().create_videos()

    # Steps 2–4 — train, evaluate, visualise (see tracking.ipynb)
    # dataset = fo.load_dataset("mot20")
    # session = fo.launch_app(dataset)
    # session.wait()
