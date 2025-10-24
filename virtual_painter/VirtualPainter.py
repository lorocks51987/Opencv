import cv2
import numpy as np
import HandTrackingModule as htm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Cabeçalhos
folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]
drawColor = (255, 100, 0)  # azul mais escuro como cor inicial
brushThickness = 15
eraseThickness = 50


# Câmera
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Configurar janela para tela cheia
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

detector = htm.handDetector(detectionCon=0.85)
xp,yp = 0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) == 21:  # só processa se a mão estiver completa
        x1, y1 = lmList[8][1:]   # ponta do dedo indicador
        x2, y2 = lmList[12][1:]  # ponta do dedo médio

        fingers = detector.fingersUp()  # agora seguro

        # Modo Seleção: indicador e médio levantados
        if fingers[1] and fingers[2]:
            print("Modo Seleção")
            if y1 < 125:
                h, w = img.shape[:2]
                # Ajusta as coordenadas dos botões baseado na largura da tela
                btn_width = w // 4
                if 0 < x1 < btn_width:
                    header = overlayList[0]
                    drawColor = (0, 255, 0)     # verde
                    xp, yp = 0, 0  # Reset das coordenadas
                elif btn_width < x1 < btn_width * 2:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)     # vermelho
                    xp, yp = 0, 0  # Reset das coordenadas
                elif btn_width * 2 < x1 < btn_width * 3:
                    header = overlayList[2]
                    drawColor = (255, 100, 0)    # azul mais escuro
                    xp, yp = 0, 0  # Reset das coordenadas
                elif btn_width * 3 < x1 < w:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)    # borracha
                    xp, yp = 0, 0  # Reset das coordenadas

            cv2.rectangle(img, (x1, y1-15), (x2, y2+15), drawColor, cv2.FILLED)


        # Modo Desenho: apenas indicador levantado
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Modo Desenho")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if(drawColor == (0,0,0)):
                cv2.line(img,(xp,yp), (x1,y1),drawColor, eraseThickness)
                cv2.line(imgCanvas,(xp,yp), (x1,y1),drawColor, eraseThickness)
            else:
                cv2.line(img,(xp,yp), (x1,y1),drawColor, brushThickness)
                cv2.line(imgCanvas,(xp,yp), (x1,y1),drawColor, brushThickness)

            xp, yp = x1, y1
        else:
            # Reset das coordenadas quando não está desenhando
            xp, yp = 0, 0

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2 .THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)




    # Coloca o cabeçalho (redimensiona para se ajustar à largura da imagem)
    h, w = img.shape[:2]
    header_resized = cv2.resize(header, (w, 125))
    img[0:125, 0:w] = header_resized
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para sair
        break
