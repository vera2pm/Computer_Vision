from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from sahi.annotation import BoundingBox, Mask

from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import pandas as pd
from src.utils import cv2_load2rgb

root_folder = "../data/MOT20/"
folder = "../data/MOT20/test/MOT20-04/"


class Preparation:
    def __init__(self, new_size, path):
        self.new_size = new_size

        self.path = path

    def convert(self, row):
        x = row[0]
        y = row[1]
        w = row[2]
        h = row[3]
        img_w, img_h = self.img_size

        # Finding midpoints
        x_centre = x + (w / 2)
        y_centre = y + (h / 2)

        x_centre = x_centre / img_w
        y_centre = y_centre / img_h
        w_scaled = w / img_w
        h_scaled = h / img_h

        # Limiting upto fix number of decimal places
        x_centre = format(x_centre, ".6f")
        y_centre = format(y_centre, ".6f")
        w_scaled = format(w_scaled, ".6f")
        h_scaled = format(h_scaled, ".6f")

        return x_centre, y_centre, w_scaled, h_scaled

    def label_preparing(self):
        path = self.path
        print("Preparing Labels")
        # path = "drive/MyDrive/"
        df = pd.read_table(f"{path}/gt/gt.txt", sep=",", header=None)
        # os.mkdir(path + "/labels/")

        cols = ["frame", "id_track", "lt_x", "lt_y", "width", "height", "score", "class", "visibility"]
        df = df.rename(columns={i: v for i, v in enumerate(cols)})
        center_coords = df[["lt_x", "lt_y", "width", "height"]].apply(self.convert, axis=1).values.tolist()
        # print(center_coords)
        df2 = pd.DataFrame(center_coords, columns=["x_center", "y_center", "width", "height"], dtype="float16")
        center_coords = df2.apply(self.resize_boxes, axis=1).values.tolist()
        df2 = pd.DataFrame(
            center_coords, columns=["x_center", "y_center", "width_scaled", "height_scaled"], dtype="float16"
        )
        df = df.merge(df2, how="left", left_index=True, right_index=True)
        print(df.info())

        df["class"] = df["class"] - 1
        df = df.sort_values("frame", ascending=True)
        for frame in df["frame"].unique():
            frame_name = str(frame)
            while len(frame_name) < 6:
                frame_name = "0" + frame_name
            # print(frame_name)
            df_yolo = df.loc[df["frame"] == frame][["class", "x_center", "y_center", "width_scaled", "height_scaled"]]
            df_yolo.to_csv(f"{path}/labels/{frame_name}.txt", index=False, header=False, sep=" ")

    def run(self):
        print("Running slicing and label preparation")
        self.tile_images_and_labels()

    def tile_images_and_labels(self):
        image_dir = os.path.join(self.path, "images")
        label_dir = os.path.join(self.path, "labels")
        os.makedirs(label_dir, exist_ok=True)
        for image_file in os.listdir(image_dir):
            if not image_file.endswith((".jpg", ".png")):
                continue
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            slices = slice_image(
                image, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2
            )
            for idx, sliced in enumerate(slices):
                out_img = sliced.image
                out_boxes = sliced.bboxes_xywh
                filename = f"{os.path.splitext(image_file)[0]}_{idx}.jpg"
                labelname = f"{os.path.splitext(image_file)[0]}_{idx}.txt"
                cv2.imwrite(os.path.join(image_dir, filename), out_img)
                with open(os.path.join(label_dir, labelname), "w") as f:
                    for box in out_boxes:
                        x_c, y_c, w, h = box
                        f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    def get_augmentations(self):
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )


def train_detector(data_path=f"../mot20.yaml"):
    model = YOLO("yolo11n.pt")
    results = model.train(data=data_path, epochs=2, device="mps", batch=8)
    print(results)
    path = model.export()
    print(f"Model exported to {path}")
    return model


def xyxy_to_yolo(xyxy, img_w, img_h):
    x_min, y_min, x_max, y_max = xyxy
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return [float(v) for v in [x_center, y_center, width, height]]


def main(video, object_detector=None):
    video_new = None
    # Load a model
    object_detector = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda:0",  # or 'cpu'
    )
    tracker = DeepSort(max_age=5, embedder="clip_ViT-B/32")
    i = 0
    for frame_path in tqdm(video):
        frame = cv2.imread(folder + frame_path)
        if video_new is None:
            height, width, layers = frame.shape
            video_new = cv2.VideoWriter(
                f"{root_folder}/Tracked_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (width, height)
            )
        results = get_sliced_prediction(
            folder + frame_path,
            detection_model=object_detector,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        bbs = []
        width = results.image.size[0]
        hight = results.image.size[1]
        for res in results.object_prediction_list:
            xywh = xyxy_to_yolo(res.bbox.to_xyxy(), width, hight)
            bbs.append((xywh, res.score.value, res.category.id))
        annotated_frame = np.array(results.image)
        # print(annotated_frame)
        tracks = tracker.update_tracks(
            bbs, frame=frame
        )  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        for track in tracks:
            # print("IMH")
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            points = np.hstack(ltrb).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # cv2_imshow(annotated_frame)
        video_new.write(annotated_frame)

    video_new.release()


if __name__ == "__main__":
    import torch

    torch.mps.empty_cache()
    # for path in ["train/MOT20-01", "train/MOT20-02"]:
    #     Preparation(new_size=(1024, 1024), path=root_folder + path).run()
    #
    # for path in ["test/MOT20-04"]:  # , "test/MOT20-06/img1", "test/MOT20-07/img1", "test/MOT20-08/img1"]:
    #     # label_preparing("./drive/MyDrive/" + path)
    #     Preparation(new_size=(1024, 1024), path=root_folder + path).image_scale()

    obj_detector = train_detector("../data/train_mot20/dataset.yaml")
    # obj_detector = YOLO("yolo11n.pt")
    obj_detector.export()
    # video = np.sort(os.listdir(folder))
    # main(video, obj_detector)
