

import torch
import cv2

def run_yolo_webcam(local_repo, weights_path):
    model = torch.hub.load(local_repo, 'custom', path=weights_path, source='local')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Run inference on the current frame (frame is a numpy array)
        results = model(frame)

        # Render results on the frame
        img_with_boxes = results.render()[0]

        # Show the frame with detections
        cv2.imshow('YOLOv5 Webcam', img_with_boxes)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run_yolo_webcam(
    local_repo='C:/Users/27622/Documents/Python/SwarmScout/yolov5',
    weights_path='C:/Users/27622/Documents/Python/SwarmScout/yolov5/yolov5s.pt'
)