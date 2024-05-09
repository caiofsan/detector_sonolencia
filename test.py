import torch
import numpy as np
import cv2

# load custom model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/best.pt', force_reload=True)

# capture image
cap = cv2.VideoCapture(0)

while cap.isOpened():
    (ret, frame) = cap.read() # A tuple that contains two values: a boolean and an array

    # Make detections ENABLED
    results = model(frame)
    cv2.imshow('YOLO Webcam Test', np.squeeze(results.render()))
    #cv2.imshow('YOLO Webcam Test', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
