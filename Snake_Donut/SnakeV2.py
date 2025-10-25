import cvzone
import cv2
import numpy as np
import math
import random
import time
from cvzone.HandTrackingModule import HandDetector
from PIL import ImageFont, ImageDraw, Image

# Configuração da câmera
cap = cv2.VideoCapture(1) # Tente '0' se '1' não funcionar
cap.set(3, 1280)
cap.set(4, 720)

# Detector de mão
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Fonte para a tela inicial (e agora para o Game Over)
try:
    font_pil_large = ImageFont.truetype("arial.ttf", 50)
    font_pil_small = ImageFont.truetype("arial.ttf", 30)
except IOError:
    print("Aviso: 'arial.ttf' não encontrada. Carregando fonte padrão.")
    font_pil_large = ImageFont.load_default()
    font_pil_small = ImageFont.load_default()


class SnakeGame:
    def __init__(self):
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.prev_head = None

        try:
            self.imgFood = cv2.imread("donut.png", cv2.IMREAD_UNCHANGED)
            self.hFood, self.wFood, _ = self.imgFood.shape
        except Exception as e:
            print(f"Erro ao carregar 'donut.png': {e}")
            # Cria um donut 'fake' caso o arquivo não exista
            self.imgFood = np.zeros((100, 100, 4), dtype=np.uint8)
            cv2.circle(self.imgFood, (50, 50), 40, (0, 0, 255, 255), -1) # Donut vermelho
            cv2.circle(self.imgFood, (50, 50), 10, (255, 255, 255, 255), -1) # Buraco
            self.hFood, self.wFood = 100, 100


        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False
        self.handToggleSeq = 0
        self.prevHandState = None

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
        self.handToggleSeq = 0
        self.prevHandState = None

    def update(self, imgMain, currentHead, fingers=None):
        
        # Lógica de Reset (mão aberta/fechada)
        if fingers is not None:
            if all(f == 0 for f in fingers): # Mão fechada
                if self.prevHandState == 'open':
                    self.handToggleSeq += 1
                self.prevHandState = 'closed'
            else: # Mão aberta
                self.prevHandState = 'open'

            if self.handToggleSeq >= 2:
                self.resetGame()

        # ----- MODIFICAÇÃO PARA TELA ESTILO GTA (COM PIL) -----
        if self.gameOver:
            # 1. Cria uma camada preta semi-transparente
            overlay = imgMain.copy()
            alpha = 0.6 # Transparência
            cv2.rectangle(overlay, (0, 0), (imgMain.shape[1], imgMain.shape[0]), (0, 0, 0), -1)
            imgMain = cv2.addWeighted(overlay, alpha, imgMain, 1 - alpha, 0)
            
            # 2. Texto principal "PERDEU!" (cvzone ainda serve para este)
            text_lost = "PERDEU!"
            cvzone.putTextRect(imgMain, text_lost, [300, 300],
                               scale=5, thickness=5, offset=20,
                               colorR=(0,0,0), colorT=(0,0,255),
                               font=cv2.FONT_HERSHEY_COMPLEX)
            
            # ----- CORREÇÃO DE ACENTOS E CENTRALIZAÇÃO (USANDO PIL) -----
            
            # 3. Converter imagem OpenCV (BGR) para PIL (RGB)
            img_pil = Image.fromarray(cv2.cvtColor(imgMain, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 4. Definir as linhas de texto
            text_line1 = "Faça o sinal de reset com a mão"
            text_line2 = "para jogar novamente"
            
            y_start = 450 # Posição Y inicial (abaixo do "PERDEU!")
            
            # 5. Calcular e desenhar Linha 1 (centralizada)
            bbox1 = draw.textbbox((0, 0), text_line1, font=font_pil_small)
            w1 = bbox1[2] - bbox1[0]
            h1 = bbox1[3] - bbox1[1]
            x1 = (imgMain.shape[1] // 2) - (w1 // 2)
            draw.text((x1, y_start), text_line1, font=font_pil_small, fill=(255, 255, 255, 255))
            
            # 6. Calcular e desenhar Linha 2 (centralizada)
            bbox2 = draw.textbbox((0, 0), text_line2, font=font_pil_small)
            w2 = bbox2[2] - bbox2[0]
            x2 = (imgMain.shape[1] // 2) - (w2 // 2)
            # Posiciona a linha 2 abaixo da linha 1, com um espaçamento de 10px
            draw.text((x2, y_start + h1 + 10), text_line2, font=font_pil_small, fill=(255, 255, 255, 255))

            # 7. Converter de volta para OpenCV (BGR)
            imgMain = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            # ---------------------------------------------------------
                               
            # 8. Score final (desenhado por último)
            cvzone.putTextRect(imgMain, f'Score Final: {self.score}', [50, 80],
                               scale=2, thickness=2, offset=5,
                               colorR=(0,0,0), colorT=(0,255,0))
            
            return imgMain # Retorna a imagem e não executa o resto da lógica
        # ----------------------------------------------


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

        # Comida
        rx, ry = self.foodPoint
        if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and ry - self.hFood // 2 < cy < ry + self.hFood // 2:
            self.randomFoodLocation()
            self.allowed_length += 50
            self.score += 1

        # Cobra verde neon
        snake_color = (0, 255, 0)
        for i, point in enumerate(self.points):
            if i != 0:
                cv2.line(imgMain, self.points[i-1], point, snake_color, 20)
        if self.points:
            cv2.circle(imgMain, self.points[-1], 20, snake_color, cv2.FILLED)

        # Comida pulsando
        scale = 1 + 0.05 * math.sin(time.time()*5)
        # Lidar com o caso de a imagem da comida ser redimensionada para 0
        wFoodScaled = int(self.wFood*scale)
        hFoodScaled = int(self.hFood*scale)
        if wFoodScaled > 0 and hFoodScaled > 0:
            food = cv2.resize(self.imgFood, (wFoodScaled, hFoodScaled))
            imgMain = cvzone.overlayPNG(imgMain, food, (rx - food.shape[1]//2, ry - food.shape[0]//2))

        # Score verde neon
        cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80],
                           scale=2, thickness=2, offset=5,
                           colorR=(0,0,0), colorT=(0,255,0))

        # Colisão com o corpo (distância da cabeça para cada ponto)
        if len(self.points) > 20:
            head_x, head_y = self.points[-1]
            for pt in self.points[:-20]:
                if math.hypot(head_x - pt[0], head_y - pt[1]) < 30:
                    self.gameOver = True
                    break

        return imgMain

# Instância do jogo
game = SnakeGame()

# Loop principal do jogo
while True:
    success, img = cap.read()
    if not success:
        print("Erro ao ler frame da câmera. Verifique a conexão.")
        break
        
    # img = cv2.flip(img, 1) # Inverte a imagem horizontalmente
    hands, img = detector.findHands(img, flipType=False) # flipType=False pois já invertemos

    if hands:
        lmList = hands[0]['lmList']
        fingers = detector.fingersUp(hands[0])
        # Pega a ponta do dedo indicador
        pontIndex = lmList[8][0:2] 
        cv2.circle(img, tuple(pontIndex), 10, (0,255,0), cv2.FILLED) # Círculo verde na ponta do dedo
        
        # Atualiza o jogo com a posição do dedo
        img = game.update(img, pontIndex, fingers)
    else:
        # Tela inicial (sem mão detectada)
        
        # Converte para PIL para desenhar com acentos
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        text = "Mostre a mão para começar"
        bbox = draw.textbbox((0,0), text, font=font_pil_large)
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        x = img.shape[1]//2 - w//2
        y = img.shape[0]//2 - h//2
        
        # Fundo preto para o texto
        draw.rectangle([x-10, y-10, x+w+10, y+h+10], fill=(0,0,0,255)) 
        # Texto verde neon
        draw.text((x,y), text, font=font_pil_large, fill=(0,255,0,255))
        
        # Converte de volta para OpenCV
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Exibe a imagem final
    cv2.imshow("Snake Game", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: # 'q' ou ESC para sair
        break

cap.release()
cv2.destroyAllWindows()