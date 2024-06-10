import cv2
import torch
import numpy as np

# carregando o modelo YOLO personalizado
model = torch.hub.load('ultralytics/yolov5', 
                       'custom', 
                       path='drowsiness_model/exp2/best.pt')

# propriedades da janela da webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)

# Iniciando webcam
if not webcam.isOpened():
    print("Não foi possível abrir a webcam.")
    exit()
while True:
    ret, img = webcam.read() # fazendo a leitura de um frame
    detection = model(img) # o modelo processa o frame e guarda o resultado

    cv2.imshow('Teste de captura', 
               np.squeeze(detection.render())) # o resultado é renderizado

    if cv2.waitKey(1) == ord('q'): # Aperte Q para encerrar a aplicação
        break

webcam.release()
cv2.destroyAllWindows()