import cv2
import numpy as np
from collections import defaultdict, deque
import supervision as sv
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO("yolov8x.pt")

# Class for transforming points based on perspective transformation
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# Video capture and settings
cap = cv2.VideoCapture("input.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer setup
video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize ByteTrack and other annotators
byte_track = sv.ByteTrack(frame_rate=fps, track_thresh=0.7)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=fps * 2, position=sv.Position.BOTTOM_CENTER)

# Source and target points for perspective transformation
SOURCE = np.array([[353, 310], [924, 310], [1908, 688], [-715, 688]])
TARGET_WIDTH, TARGET_HEIGHT = 57, 98
TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])

# Initialize view transformer and coordinate tracking
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
polygon_zone = sv.PolygonZone(polygon=SOURCE)
coordinates = defaultdict(lambda: deque(maxlen=fps * 2))

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run detection and filter results
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > 0.5]
    detections = detections[polygon_zone.trigger(detections)]
    detections = detections.with_nms(threshold=0.7)
    detections = byte_track.update_with_detections(detections=detections)

    # Transform points and update coordinates
    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    points = view_transformer.transform_points(points=points).astype(int)
    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)

    # Create labels based on speed calculation
    labels = []
    for tracker_id in detections.tracker_id:
        if len(coordinates[tracker_id]) < fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            start_y, end_y = coordinates[tracker_id][-1], coordinates[tracker_id][0]
            distance = abs(start_y - end_y)
            time_elapsed = len(coordinates[tracker_id]) / fps
            speed = (distance / time_elapsed) * 3.6  # Convert from m/s to km/h
            labels.append(f"#{tracker_id} {int(speed)} km/h")

    # Annotate the frame with detections, traces, and labels
    annotated_frame = frame.copy()
    annotated_frame = cv2.polylines(annotated_frame, [SOURCE], isClosed=True, color=(100, 255, 0), thickness=2)
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Write the annotated frame to the output video
    video_writer.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()
