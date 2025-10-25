import cvzone
import cv2
import numpy as np
import math
import random
import time
from cvzone.HandTrackingModule import HandDetector
from PIL import ImageFont, ImageDraw, Image

# --- Configuração da câmera ---
cap = cv2.VideoCapture(1)  # tente 0 se não funcionar
cap.set(3, 1280)
cap.set(4, 720)

# --- Detector de mãos ---
detector = HandDetector(detectionCon=0.8, maxHands=1)

# --- Fonte ---
try:
    font_large = ImageFont.truetype("arial.ttf", 50)
    font_small = ImageFont.truetype("arial.ttf", 30)
except:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

# --- Carregar GIF da maçã ---
apple_gif = Image.open("enchanted_apple.gif")
apple_frames = []
try:
    while True:
        frame = apple_gif.convert("RGBA").resize((75, 75))  # já redimensiona aqui
        apple_frames.append(frame.copy())
        apple_gif.seek(apple_gif.tell() + 1)
except EOFError:
    pass

num_apple_frames = len(apple_frames)
apple_frame_index = 0

class SnakeGame:
    def __init__(self):
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.prev_head = None

        # Donut
        try:
            self.imgFood = cv2.imread("donut.png", cv2.IMREAD_UNCHANGED)
            self.hFood, self.wFood, _ = self.imgFood.shape
        except:
            self.imgFood = np.zeros((75,75,4), dtype=np.uint8)
            cv2.circle(self.imgFood, (37,37), 35, (0,0,255,255), -1)
            cv2.circle(self.imgFood, (37,37), 10, (255,255,255,255), -1)
            self.hFood, self.wFood = 75,75

        self.foodPoint = 0,0
        self.randomFoodLocation()

        # Maçã
        self.apple_active = False
        self.apple_pos = (0,0)
        self.apple_scale = 75

        self.powerup_active = False
        self.powerup_timer = 0

        self.score = 0
        self.gameOver = False
        self.handToggleSeq = 0
        self.prevHandState = None

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def spawnApple(self):
        self.apple_active = True
        self.apple_pos = random.randint(100, 1000), random.randint(100, 600)

    def resetGame(self):
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.prev_head = None
        self.randomFoodLocation()
        self.apple_active = False
        self.powerup_active = False
        self.powerup_timer = 0
        self.score = 0
        self.gameOver = False
        self.handToggleSeq = 0
        self.prevHandState = None

    def update(self, imgMain, currentHead, fingers=None):
        global apple_frame_index

        # Reset só se gameOver
        if self.gameOver and fingers is not None:
            if all(f==0 for f in fingers):
                if self.prevHandState == 'open':
                    self.handToggleSeq +=1
                self.prevHandState = 'closed'
            else:
                self.prevHandState = 'open'
            if self.handToggleSeq >=2:
                self.resetGame()

        # Tela GameOver
        if self.gameOver:
            overlay = imgMain.copy()
            cv2.rectangle(overlay,(0,0),(imgMain.shape[1],imgMain.shape[0]),(0,0,0),-1)
            imgMain = cv2.addWeighted(overlay,0.6,imgMain,0.4,0)
            cvzone.putTextRect(imgMain, "PERDEU!", [300,300], scale=5, thickness=5, offset=20, colorR=(0,0,0), colorT=(0,0,255), font=cv2.FONT_HERSHEY_COMPLEX)
            img_pil = Image.fromarray(cv2.cvtColor(imgMain, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            text1 = "Faça o sinal de reset com a mão"
            text2 = "para jogar novamente"
            y_start = 450
            bbox1 = draw.textbbox((0,0), text1, font=font_small)
            w1 = bbox1[2]-bbox1[0]; h1 = bbox1[3]-bbox1[1]; x1 = imgMain.shape[1]//2 - w1//2
            draw.text((x1,y_start), text1, font=font_small, fill=(255,255,255,255))
            bbox2 = draw.textbbox((0,0), text2, font=font_small)
            w2 = bbox2[2]-bbox2[0]; x2 = imgMain.shape[1]//2 - w2//2
            draw.text((x2,y_start+h1+10), text2, font=font_small, fill=(255,255,255,255))
            imgMain = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cvzone.putTextRect(imgMain, f'Score Final: {self.score}', [50,80], scale=2, thickness=2, offset=5, colorR=(0,0,0), colorT=(0,255,0))
            return imgMain

        if self.prev_head is None:
            self.prev_head = currentHead

        px, py = self.prev_head
        cx, cy = currentHead
        self.points.append([cx,cy])
        distance = math.hypot(cx-px, cy-py)
        self.lengths.append(distance)
        self.current_length += distance
        self.prev_head = cx,cy

        while self.current_length > self.allowed_length:
            self.current_length -= self.lengths[0]
            self.lengths.pop(0)
            self.points.pop(0)

        # Comida donut
        fx, fy = self.foodPoint
        if fx - self.wFood//2 < cx < fx+self.wFood//2 and fy - self.hFood//2 < cy < fy+self.hFood//2:
            self.randomFoodLocation()
            self.allowed_length += 30
            self.score += 1

        # Cobra verde neon
        snake_color = (0,255,0)
        for i in range(1,len(self.points)):
            cv2.line(imgMain, self.points[i-1], self.points[i], snake_color, 20)
        if self.points:
            cv2.circle(imgMain, self.points[-1],20,snake_color,cv2.FILLED)

        # Donut pulsando
        scale = 1 + 0.05*math.sin(time.time()*5)
        wFoodScaled = int(self.wFood*scale)
        hFoodScaled = int(self.hFood*scale)
        if wFoodScaled>0 and hFoodScaled>0:
            food = cv2.resize(self.imgFood,(wFoodScaled,hFoodScaled))
            imgMain = cvzone.overlayPNG(imgMain, food, (fx-wFoodScaled//2, fy-hFoodScaled//2))

        # Spawn maçã
        if not self.apple_active and self.score >= 15:
            if random.randint(0,10) == 1:  # 1 chance em 6 por frame
                self.spawnApple()

        # Maçã animada
        if self.apple_active and num_apple_frames>0:
            frame = apple_frames[apple_frame_index]
            apple_frame_index = (apple_frame_index +1)%num_apple_frames
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA)
            ax, ay = self.apple_pos
            imgMain = cvzone.overlayPNG(imgMain, frame_cv, (ax-37, ay-37))  # já é 75x75

            # Colisão maçã
            hx, hy = self.points[-1]
            if abs(hx-ax)<30 and abs(hy-ay)<30:
                self.apple_active = False
                self.score += 2
                self.powerup_active = True
                self.powerup_timer = time.time()

        # Power-up info
        if self.powerup_active:
            elapsed = time.time() - self.powerup_timer
            if elapsed > 5:
                self.powerup_active = False
            else:
                cvzone.putTextRect(imgMain, f'IMORTALIDADE ({5-int(elapsed)}s)', [50,130], scale=2, thickness=2, offset=5, colorR=(0,0,0), colorT=(0,255,0))

        # Score
        cvzone.putTextRect(imgMain, f'Score: {self.score}', [50,80], scale=2, thickness=2, offset=5, colorR=(0,0,0), colorT=(0,255,0))

        # Colisão corpo (não colide se powerup ativo)
        if len(self.points)>20 and not self.powerup_active:
            hx, hy = self.points[-1]
            for pt in self.points[:-20]:
                if math.hypot(hx-pt[0], hy-pt[1])<30:
                    self.gameOver = True
                    break

        return imgMain

# --- Instancia ---
game = SnakeGame()

# --- Loop principal ---
while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        fingers = detector.fingersUp(hands[0])
        pontIndex = lmList[8][0:2]
        cv2.circle(img, tuple(pontIndex),10,(0,255,0),cv2.FILLED)
        img = game.update(img, pontIndex, fingers)
    else:
        # Tela inicial
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        text = "Mostre a mão para começar"
        bbox = draw.textbbox((0,0), text, font=font_large)
        w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
        x = img.shape[1]//2 - w//2
        y = img.shape[0]//2 - h//2
        draw.rectangle([x-10,y-10,x+w+10,y+h+10], fill=(0,0,0,255))
        draw.text((x,y), text, font=font_large, fill=(0,255,0,255))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q') or key==27:
        break

cap.release()
cv2.destroyAllWindows()
