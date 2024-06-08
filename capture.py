import cv2
import torch
import numpy as np

# webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# modelo personalizado
model = torch.hub.load('ultralytics/yolov5', 'custom', path='drowsiness_model/exp2/last.pt', force_reload=True)
# classes
classes = ["acordado", "sonolento"]

while True:
    ret, img = cap.read()
    img = [img]
    results = model(img)

    cv2.imshow('Teste de captura', np.squeeze(results.render()))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()