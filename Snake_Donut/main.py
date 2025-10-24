import cvzone
import cv2
import numpy as np
import math
import random
import time
from cvzone.HandTrackingModule import HandDetector

# Configura a câmera
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Detector de mãos
detector = HandDetector(detectionCon=0.8, maxHands=1)

class SnakeGame:
    def __init__(self):
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.prev_head = None

        self.imgFood = cv2.imread("donut.png", cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False
        self.handClosedTime = None  # tempo que a mão está fechada

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def resetGame(self):
        self.gameOver = False
        self.score = 0
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.prev_head = None
        self.randomFoodLocation()
        self.handClosedTime = None

    def update(self, imgMain, currentHead, fingers=None):
        # Reinício automático se mão fechada >=1s
        if fingers is not None:
            if all(f == 0 for f in fingers):
                if self.handClosedTime is None:
                    self.handClosedTime = time.time()
                elif time.time() - self.handClosedTime >= 1:
                    self.resetGame()
            else:
                self.handClosedTime = None

        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", (500, 300), scale=3, thickness=3, offset=20)
            cvzone.putTextRect(imgMain, f"Score: {self.score}", (550, 400), scale=2, thickness=2, offset=10)
            return imgMain

        # Inicializa prev_head
        if self.prev_head is None:
            self.prev_head = currentHead

        px, py = self.prev_head
        cx, cy = currentHead

        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.current_length += distance
        self.prev_head = cx, cy

        if self.current_length > self.allowed_length:
            for i, length in enumerate(self.lengths):
                self.current_length -= length
                self.lengths.pop(i)
                self.points.pop(i)
                if self.current_length < self.allowed_length:
                    break

        # Verifica se comeu a comida
        rx, ry = self.foodPoint
        if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and ry - self.hFood // 2 < cy < ry + self.hFood // 2:
            self.randomFoodLocation()
            self.allowed_length += 50
            self.score += 1
            print(self.score)

        # Desenha a cobra
        for i, point in enumerate(self.points):
            if i != 0:
                cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
        if self.points:
            cv2.circle(imgMain, self.points[-1], 20, (0, 255, 0), cv2.FILLED)

        # Desenha a comida
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        # Score
        cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80], scale=3, thickness=3, offset=10)

        # Checa colisão com a própria cobra
        if len(self.points) > 4 and distance > 5:
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            if -1 < minDist <= 1:
                print("Game Over")
                self.gameOver = True

        return imgMain

# Inicializa o jogo
game = SnakeGame()

# Loop principal
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        fingers = detector.fingersUp(hands[0])
        pontIndex = lmList[8][0:2]
        img = game.update(img, pontIndex, fingers)

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
