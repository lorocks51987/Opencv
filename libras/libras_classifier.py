import cv2
import numpy as np
import tensorflow as tf
import time
from cvzone.HandTrackingModule import HandDetector 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carregar modelo Keras
model = tf.keras.models.load_model("Model/keras_model.h5")

# Carregar labels
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Configurações da câmera
cap = None
available_cameras = []

for i in range(5):
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        ret, test_img = test_cap.read()
        if ret and test_img is not None:
            available_cameras.append(i)
        test_cap.release()

if not available_cameras:
    print("Erro: Nenhuma câmera encontrada!")
    exit()

camera_order = sorted(available_cameras, reverse=True)
FORCE_CAMERA = 1  # Mude para o índice da câmera que quer usar
if FORCE_CAMERA is not None and FORCE_CAMERA in available_cameras:
    cap = cv2.VideoCapture(FORCE_CAMERA)
else:
    for i in camera_order:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, test_img = cap.read()
            if ret and test_img is not None:
                break
            cap.release()

if cap is None or not cap.isOpened():
    print("Erro: Nenhuma câmera encontrada!")
    exit()

detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success or img is None:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Recorte da mão com offset
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        # Criar fundo branco
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Manter proporção
        h_crop, w_crop = imgCrop.shape[:2]
        aspectRatio = h_crop / w_crop

        if aspectRatio > 1:  # altura maior
            newH = imgSize
            newW = int(w_crop * imgSize / h_crop)
        else:  # largura maior
            newW = imgSize
            newH = int(h_crop * imgSize / w_crop)

        imgResize = cv2.resize(imgCrop, (newW, newH))
        xOffset = (imgSize - newW) // 2
        yOffset = (imgSize - newH) // 2
        imgWhite[yOffset:yOffset+newH, xOffset:xOffset+newW] = imgResize

        # Normalizar e fazer previsão
        img_input = imgWhite / 255.0
        img_input = np.expand_dims(img_input, axis=0)  # shape (1, 300, 300, 3)
        prediction = model.predict(img_input)
        index = np.argmax(prediction)
        print(prediction, index)

        # Desenhar resultado na imagem final
        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Webcam", imgOutput)
    key = cv2.waitKey(1)
    if key == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
