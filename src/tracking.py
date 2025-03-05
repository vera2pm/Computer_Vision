import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import pandas as pd
from src.utils import cv2_load2rgb

folder = ""


def convert(row, img_size=(1920, 1080)):
    x = row[0]
    y = row[1]
    w = row[2]
    h = row[3]
    img_w, img_h = img_size

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


def label_preparing(path):
    # path = "drive/MyDrive/"
    df = pd.read_table(f"{path}/gt/gt.txt", sep=",", header=None)
    # os.mkdir(path + "/labels/")

    cols = ["frame", "id_track", "lt_x", "lt_y", "width", "height", "score", "class", "visibility"]
    df = df.rename(columns={i: v for i, v in enumerate(cols)})
    center_coords = df[["lt_x", "lt_y", "width", "height"]].apply(convert, axis=1).values
    print(center_coords)
    df["x_center"] = [coords[0] for coords in center_coords]
    df["y_center"] = [coords[1] for coords in center_coords]
    df["width"] = [coords[2] for coords in center_coords]
    df["height"] = [coords[3] for coords in center_coords]
    df["class"] = df["class"] - 1
    df = df.sort_values("frame", ascending=True)
    for frame in df["frame"].unique():
        frame_name = str(frame)
        while len(frame_name) < 6:
            frame_name = "0" + frame_name
        # print(frame_name)
        df_yolo = df.loc[df["frame"] == frame][["class", "x_center", "y_center", "width", "height"]]
        # tfile = open(f"{path}/labels/{frame_name}.txt", 'a')
        df_yolo.to_csv(f"{path}/labels/{frame_name}.txt", index=False, header=False, sep=" ")
        # tfile.write(row.to_string())
        # tfile.close()


def train_detector():
    model = YOLO("yolov8n.pt")
    results = model.train(data="mot20.yaml", epochs=100, imgsz=640)
    print(results)


def main(video):
    video_new = None
    # Load a model
    object_detector = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=5, embedder="clip_ViT-B/32")
    for i, frame_path in tqdm(enumerate(video)):
        frame = cv2_load2rgb(folder + frame_path)
        if video_new is None:
            height, width, layers = frame.shape
            video_new = cv2.VideoWriter(
                "drive/MyDrive/test/MOT20-04/Tracked_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
            )
        results = object_detector(frame, classes=[0], verbose=False)
        bbs = []
        for res in results[0].boxes:
            bbs.append((res.xywh.cpu().numpy()[0], res.conf.cpu().numpy()[0], res.cls.cpu().numpy()[0]))
        annotated_frame = results[0].plot()
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
    # break
