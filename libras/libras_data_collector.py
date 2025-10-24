import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "data/H"
if not os.path.exists(folder):
    os.makedirs(folder)

counter = 0

while True:
    success, img = cap.read()
    if not success:
        break
        
    hands, img = detector.findHands(img)

    # frame = cv2.flip(frame, 1)  # espelha a câmera horizontalmente
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

        # Centralizar no fundo branco
        xOffset = (imgSize - newW) // 2
        yOffset = (imgSize - newH) // 2
        imgWhite[yOffset:yOffset+newH, xOffset:xOffset+newW] = imgResize

        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Webcam", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Imagem {counter} salva!")
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
