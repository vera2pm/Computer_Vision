from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


def main(video):
    # Load a model
    object_detector = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=5, embedder="clip_ViT-B/32")
    for frame in video:
        results = object_detector.track(frame)
        bbs = results[0].boxes.xywh  # .cpu()
        tracks = tracker.update_tracks(
            bbs, frame=frame
        )  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
