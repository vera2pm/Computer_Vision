import os
import sys

import pandas as pd
from sahi.utils.ultralytics import download_yolo11n_model

# from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path

sys.path.append("../")
from deep_sort_realtime.deepsort_tracker import DeepSort
from boxmot import BotSort
from ultralytics import YOLO
from ultralytics.utils.ops import xyxy2ltwh, xywh2ltwh, ltwh2xywh
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import fiftyone as fo
import fiftyone.utils.video as fouv
from fiftyone import ViewField as F

print(f"Your FO version is: {fo.__version__}")

root_folder = "../data/"
mv_dir = os.path.join(root_folder, "mot20_mvs")
model_path = "models/yolo11n.pt"
download_yolo11n_model(model_path)


def create_videos():
    for movi in ["MOT20/train/MOT20-01/"]:  # "MOT20/train/MOT20-05/", "MOT20/test/MOT20-06/",
        print(movi)
        mov_in = os.path.join(root_folder, movi, f"img1/%06d.jpg")
        name = movi.split("/")[2]
        mov_out = os.path.join(mv_dir, f"movi_{name}.mp4")
        fouv.reencode_video(
            mov_in,
            mov_out,
            verbose=False,
        )


GT_COLUMNS = ["frame", "id", "x0", "y0", "w", "h", "flag", "class", "visibility"]
GT_CLASSES = ["na", "person", "na", "na", "na", "na", "vehicle", "person", "na", "na", "na", "occluded", "na", "crowd"]


def save_labels_from_sample(sample):
    # for sample in dataset[:1]:
    # Match image filename root
    video_stem = os.path.splitext(os.path.basename(sample.filepath))[0]
    print(video_stem)
    os.makedirs(f"../data/train_mot20/{video_stem}/labels")

    for i in range(len(sample.frames)):
        frame_number = i + 1
        f = sample.frames[frame_number]
        output_file = open(f"../data/train_mot20/{video_stem}/labels/{frame_number:06d}.txt", "w")
        for det in f.gt.detections:
            label = det.label  # string
            class_id = 0  # adjust this based on label if multi-class
            ltwhn = np.array(det.bounding_box)
            # print(label)

            # Convert from [x, y, w, h] (normalized) to YOLO [x_center, y_center, w, h]
            xc, yc, w, h = list(ltwh2xywh(ltwhn))
            output_file.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        output_file.close()


def add_ground_truth():
    dataset = fo.load_dataset("mot20")
    view = dataset.skip(2)
    frames = view.to_frames(sample_frames=True)
    df = pd.read_table(
        os.path.join(root_folder, "MOT20/train/MOT20-03/gt/gt.txt"),
        header=None,
        sep=",",
    )
    view = dataset.last()
    df.columns = GT_COLUMNS
    imw = view.metadata.frame_width
    imh = view.metadata.frame_height
    df = df.sort_values(by="frame")
    frames = df["frame"].astype("int").unique()

    for i in tqdm(frames):
        f = view.frames[int(i)]
        try:
            if f.gt is None:
                f.gt = fo.Detections()
        except AttributeError:
            f["gt"] = fo.Detections()
        f.gt = fo.Detections()
        df_frame = df.loc[df.frame == i]
        for i, row in df_frame.iterrows():
            track_id = row["id"]
            label = GT_CLASSES[int(row["class"])]
            ltwh = np.array(row[["x0", "y0", "w", "h"]])
            ltwhn = np.array([ltwh[0] / imw, ltwh[1] / imh, ltwh[2] / imw, ltwh[3] / imh])
            det = fo.Detection(label=label, bounding_box=ltwhn, index=track_id)
            f.gt.detections.append(det)
        f.save()


def create_dataset():
    # create_videos()
    # dataset = fo.load_dataset("mot20")
    # dataset.delete()
    dataset = fo.Dataset.from_dir(dataset_dir=mv_dir, dataset_type=fo.types.VideoDirectory)
    dataset.ensure_frames()
    dataset.name = "mot20"
    dataset.persistent = True
    sample = dataset.first()
    print(sample)


def tracking_with_sliced_prediction(view):
    frames = view.to_frames(sample_frames=True)
    imw = view.first().metadata.frame_width
    imh = view.first().metadata.frame_height

    object_detector = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.3,
        device="mps",  # or 'cpu'
    )
    # tracker = DeepSort(max_age=5, embedder="clip_ViT-B/32")
    # tracker.device = "mps"
    tracker = BotSort(reid_weights=Path("osnet_x0_25_msmt17.pt"), device="mps", half=True)

    for i, frame_obj in tqdm(enumerate(frames), total=len(frames)):
        f = view.first().frames[i + 1]
        print(len(f.gt.detections))
        try:
            if f.bot_sort is None:
                f.bot_sort = fo.Detections()
        except AttributeError:
            f["bot_sort"] = fo.Detections()
        f.bot_sort = fo.Detections()
        # Get sliced predictions
        results = get_sliced_prediction(
            frame_obj.filepath,
            detection_model=object_detector,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        # bbs = []
        # for res in results.object_prediction_list:
        #     ltwh = xyxy2ltwh(np.array(res.bbox.to_xyxy()))
        #     bbs.append((ltwh, res.score.value, res.category.id))
        # print(np.array(bbs))
        bbs = []
        for res in results.object_prediction_list:
            xyxy = np.array(res.bbox.to_xyxy())
            bbs.append([*xyxy, res.score.value, res.category.id])
        bbs = np.array(bbs)
        print(len(bbs))

        frame = cv2.imread(frame_obj.filepath)
        tracks = tracker.update(bbs, frame)
        # tracks = tracker.update_tracks(
        #     bbs, frame=frame
        # )  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        # print(tracks)
        for track in tracks:
            # ltwh = track.to_ltwh()
            # ltwhn = np.array([ltwh[0] / imw, ltwh[1] / imh, ltwh[2] / imw, ltwh[3] / imh])
            # det = fo.Detection(label="person", bounding_box=ltwhn, index=track_id, confidence=track.det_conf)
            track_id = track[4]
            try:
                label = GT_CLASSES[int(track[6])]
            except:
                label = "None"
            det_conf = track[5]
            ltwh = xyxy2ltwh(track[:4])
            ltwhn = np.array([ltwh[0] / imw, ltwh[1] / imh, ltwh[2] / imw, ltwh[3] / imh])
            det = fo.Detection(label=label, bounding_box=ltwhn, index=track_id, confidence=det_conf)
            f.bot_sort.detections.append(det)
            f.save()


def tracking_yolo(view, model, tag):
    mov_path = view.first().filepath
    results = model.track(mov_path, device="mps", stream=True, imgsz=1280, conf=0.4)

    for frm, res in tqdm(enumerate(results), total=len(view.first().frames)):
        f = view.first().frames[frm + 1]
        try:
            f[tag] = fo.Detections()
        except AttributeError:
            f[tag] = fo.Detections()

        is_person = res.boxes.cls.cpu().numpy() == 0
        for bb, id, conf in zip(
            res.boxes.xyxyn.cpu().numpy()[is_person],
            res.boxes.id.cpu().numpy()[is_person],
            res.boxes.conf.cpu().numpy()[is_person],
        ):
            bb = xyxy2ltwh(bb)
            det = fo.Detection(label="person", bounding_box=bb, index=id, confidence=conf)
            f[tag].detections.append(det)
            f.save()


def save_results(dataset_or_view, tag):
    # The directory to which to write the annotated media
    output_dir = mv_dir + "_result_" + tag
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # The list of `Label` fields containing the labels that you wish to render on
    # the source media (e.g., classifications or detections)
    label_fields = [f"frames.{tag}"]

    # Render the labels!
    dataset_or_view.draw_labels(output_dir, label_fields=label_fields)


def main():
    # create_dataset()
    # add_ground_truth()

    dataset = fo.load_dataset("mot20")
    # save_labels_from_sample(dataset.last())

    view = dataset[1:2]
    # model = YOLO("yolo11s.pt")  # yolov8s-worldv2.pt  , yolo11s.pt
    model = YOLO("runs/detect/train/weights/best.pt")
    tracking_yolo(view, model, "yolo11_trained_10")
    # tracking_with_sliced_prediction(view)

    save_results(view, "yolo11_trained_10")


if __name__ == "__main__":
    # create_dataset()
    # add_ground_truth()

    dataset = fo.load_dataset("mot20")
    # save_labels_from_sample(dataset.last())

    session = fo.launch_app(dataset)

    session.wait()
    # print(view.first().filepath)
    # print(view.count)
    # save_results(view, "yolo11_trained_10")
