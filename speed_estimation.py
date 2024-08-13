from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8x.pt")


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


cap = cv2.VideoCapture("test_2.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Video writer
video_writer = cv2.VideoWriter(
    "speed_estimation.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

byte_track = sv.ByteTrack(frame_rate=fps, track_thresh=0.7)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER,
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness, trace_length=fps * 2, position=sv.Position.BOTTOM_CENTER
)


# Source and target points: this is unique to each video
SOURCE = np.array([[353, 310], [924, 310], [1908, 688], [-715, 688]])

TARGET_WIDTH = 57
TARGET_HEIGHT = 98

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

polygon_zone = sv.PolygonZone(polygon=SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
coordinates = defaultdict(lambda: deque(maxlen=fps * 2))

while True:
    success, frame = cap.read()
    if not success:
        break

    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > 0.5]
    detections = detections[polygon_zone.trigger(detections)]
    detections = detections.with_nms(threshold=0.7)
    detections = byte_track.update_with_detections(detections=detections)

    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    points = view_transformer.transform_points(points=points).astype(int)

    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)

    labels = []
    for tracker_id in detections.tracker_id:
        if len(coordinates[tracker_id]) < fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / fps
            speed = distance / time * 3.6
            labels.append(f"#{tracker_id} {int(speed)} km/h")

    annotated_frame = frame.copy()
    annotated_frame = cv2.polylines(
        annotated_frame, [SOURCE], isClosed=True, color=(100, 255, 0), thickness=2
    )
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    video_writer.write(annotated_frame)

    cv2.imshow("frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
