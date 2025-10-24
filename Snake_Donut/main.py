import cvzone
import cv2
import numpy as np
import math
import random
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

class SnakeGame:
    def __init__(self):
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.prev_head = 0, 0

        self.imgFood = cv2.imread("donut.png", cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):

        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", (500, 300), scale=3, thickness=3, offset=20)
            cvzone.putTextRect(imgMain, "Score: " + str(self.score), (550, 400), scale=2, thickness=2, offset=10)

        else:
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

            # Check if snake ate the Food
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
            ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowed_length += 50
                self.score += 1
                print(self.score)

            # draw snake
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
            if self.points:
                cv2.circle(imgMain, self.points[-1], 20, (0, 255, 0), cv2.FILLED)


            # draw food
            rx, ry = self.foodPoint
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))
            cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80],scale=3, thickness=3, offset=10)

            #check for collision with itself
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
            if len(self.points) > 4 and distance > 5:  # só se tiver se movido um pouco
                pts = np.array(self.points[:-2], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
                minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
                if -1 < minDist <= 1:
                    print("Game Over")
                    self.gameOver = True
                    self.points = []
                    self.lengths = []
                    self.current_length = 0
                    self.allowed_length = 150
                    self.prev_head = 0, 0
                    self.score = 0
                    self.randomFoodLocation()

        return imgMain


def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
    imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
    imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB
    imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
    maskBGRInv = cv2.bitwise_not(maskBGR)
    imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv

    imgBack = cv2.bitwise_and(imgBack, imgMaskFull2)
    imgBack = cv2.bitwise_or(imgBack, imgMaskFull)

    return imgBack


game = SnakeGame()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        Lmlist = hands[0]['lmList']
        pontIndex = Lmlist[8][0:2]
        img = game.update(img, pontIndex)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        game.gameOver = False
    if key == ord('q') or key == 27:  # 27 = ESC
        break
