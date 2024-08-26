import os
import random

import cv2
from ultralytics import YOLO

from object_tracker import Tracker


video_path = 'MAH03703.mp4'
video_out_path = 'out.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            # Define the position for the track_id text. This should be exactly at the top of the rectangle.
            text_position = (int(x1), int(y1) - 2)  # Slightly offset to avoid overlap

            # Define the font, scale, and color for the text.
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = colors[track_id % len(colors)]

            # Calculate the text size to adjust the position if necessary
            (text_width, text_height), _ = cv2.getTextSize(f'ID: {track_id}', font, font_scale, font_thickness)

            # Adjust position to make sure the text is within the frame and not overlapping with the rectangle
            text_position = (int(x1), max(int(y1) - 5, text_height))

            # Put the track_id text on the frame.
            cv2.putText(frame, f'ID: {track_id}', text_position, font, font_scale, text_color, font_thickness)

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()