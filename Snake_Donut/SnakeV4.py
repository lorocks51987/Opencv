import cvzone
import cv2
import numpy as np
import math
import random
import time
from cvzone.HandTrackingModule import HandDetector
from PIL import ImageFont, ImageDraw, Image

# --- Constantes ---
# Configurações da Câmera e Tela
CAMERA_ID = 1
LARGURA_TELA = 1280
ALTURA_TELA = 720
DETECTION_CON = 0.8
MAX_HANDS = 1

# Configurações de Fonte
FONTE_GRANDE_TAM = 50
FONTE_PEQUENA_TAM = 30
CAMINHO_FONTE = "arial.ttf"

# Configurações dos Itens
TAMANHO_MACA = 75
TAMANHO_DONUT = 75
TAMANHO_POTION = 75                 # <--- NOVO
ARQUIVO_MACA = "enchanted_apple.gif"
ARQUIVO_DONUT = "donut.png"
ARQUIVO_POTION = "Potion.png"       # <--- NOVO
ARQUIVO_HIGHSCORE = "highscore.txt"

# Configurações do Jogo
COMPRIMENTO_INICIAL = 150
PONTOS_POR_COMIDA = 1
PONTOS_POR_MACA = 2
CRESCIMENTO_POR_COMIDA = 30
SCORE_PARA_MACA = 10                # Maçã aparece após 10 pontos
SCORE_PARA_POTION = 20              # Poção aparece após 20 pontos
CHANCE_MACA_SPAWN = 50              # Chance de 1 em 50 frames (menos frequente)
CHANCE_POTION_SPAWN = 75            # Chance de 1 em 75 frames (menos frequente)
DURACAO_POWERUP_S = 5
MARGEM_SPAWN = 100

# Configurações da Cobra
COR_COBRA = (0, 255, 0)
GROSSURA_COBRA = 20
GROSSURA_CABECA = 20
RAIO_PONTA_DEDO = 10
COR_PONTA_DEDO = (0, 255, 0)
RAIO_COLISAO_CORPO = 30
RAIO_COLISAO_COMIDA = 30
DISTANCIA_COLISAO_CORPO = 20

# Posições de Texto
POS_SCORE_TXT = [50, 80]
POS_POWERUP_TXT = [50, 130]
POS_GAMEOVER_TXT = [300, 300]
SCALE_SCORE_TXT = 2
SCALE_GAMEOVER_TXT = 5
OFFSET_TEXTO = 5


# --- Configuração da câmera ---
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(3, LARGURA_TELA)
cap.set(4, ALTURA_TELA)

# --- Detector de mãos ---
detector = HandDetector(detectionCon=DETECTION_CON, maxHands=MAX_HANDS)

# --- Fonte ---
try:
    font_large = ImageFont.truetype(CAMINHO_FONTE, FONTE_GRANDE_TAM)
    font_small = ImageFont.truetype(CAMINHO_FONTE, FONTE_PEQUENA_TAM)
except:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

# --- Carregar GIF da maçã ---
apple_gif = Image.open(ARQUIVO_MACA)
apple_frames = []
try:
    while True:
        frame = apple_gif.convert("RGBA").resize((TAMANHO_MACA, TAMANHO_MACA))
        apple_frames.append(frame.copy())
        apple_gif.seek(apple_gif.tell() + 1)
except EOFError:
    pass

num_apple_frames = len(apple_frames)

class SnakeGame:
    def __init__(self):
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = COMPRIMENTO_INICIAL
        self.prev_head = None
        self.apple_frame_index = 0

        # Donut
        try:
            self.imgFood = cv2.imread(ARQUIVO_DONUT, cv2.IMREAD_UNCHANGED)
            self.hFood, self.wFood, _ = self.imgFood.shape
        except:
            self.imgFood = np.zeros((TAMANHO_DONUT, TAMANHO_DONUT, 4), dtype=np.uint8)
            cv2.circle(self.imgFood, (TAMANHO_DONUT // 2, TAMANHO_DONUT // 2), 35, (0, 0, 255, 255), -1)
            cv2.circle(self.imgFood, (TAMANHO_DONUT // 2, TAMANHO_DONUT // 2), 10, (255, 255, 255, 255), -1)
            self.hFood, self.wFood = TAMANHO_DONUT, TAMANHO_DONUT

        self.foodPoint = 0, 0
        self.randomFoodLocation()

        # Maçã
        self.apple_active = False
        self.apple_pos = (0, 0)
        
        # Poção (NOVO)
        try:
            self.imgPotion = cv2.imread(ARQUIVO_POTION, cv2.IMREAD_UNCHANGED)
            self.imgPotion = cv2.resize(self.imgPotion, (TAMANHO_POTION, TAMANHO_POTION)) 
            self.hPotion, self.wPotion, _ = self.imgPotion.shape
        except Exception as e:
            print(f"Erro ao carregar {ARQUIVO_POTION}: {e}")
            # Fallback (círculo azul) se a imagem falhar
            self.imgPotion = np.zeros((TAMANHO_POTION, TAMANHO_POTION, 4), dtype=np.uint8)
            cv2.circle(self.imgPotion, (TAMANHO_POTION // 2, TAMANHO_POTION // 2), 35, (255, 0, 0, 255), -1) 
            self.hPotion, self.wPotion = TAMANHO_POTION, TAMANHO_POTION
        
        self.potion_active = False    # <--- NOVO
        self.potion_pos = (0, 0)      # <--- NOVO
        # (Fim da adição da poção)
        
        self.powerup_active = False
        self.powerup_timer = 0

        self.score = 0
        self.gameOver = False
        self.handToggleSeq = 0
        self.prevHandState = None
        self.high_score = self.load_high_score()

    def load_high_score(self):
        """Carrega o high score do arquivo."""
        try:
            with open(ARQUIVO_HIGHSCORE, 'r') as f:
                score = int(f.read())
                return score
        except (FileNotFoundError, ValueError):
            return 0

    def save_high_score(self):
        """Salva o high score atual no arquivo."""
        try:
            with open(ARQUIVO_HIGHSCORE, 'w') as f:
                f.write(str(self.high_score))
        except Exception as e:
            print(f"Erro ao salvar high score: {e}")

    def randomFoodLocation(self):
        self.foodPoint = random.randint(MARGEM_SPAWN, LARGURA_TELA - MARGEM_SPAWN), \
                         random.randint(MARGEM_SPAWN, ALTURA_TELA - MARGEM_SPAWN)

    def spawnApple(self):
        self.apple_active = True
        self.apple_pos = random.randint(MARGEM_SPAWN, LARGURA_TELA - MARGEM_SPAWN), \
                         random.randint(MARGEM_SPAWN, ALTURA_TELA - MARGEM_SPAWN)

    def spawnPotion(self): # <--- NOVO
        self.potion_active = True
        self.potion_pos = random.randint(MARGEM_SPAWN, LARGURA_TELA - MARGEM_SPAWN), \
                          random.randint(MARGEM_SPAWN, ALTURA_TELA - MARGEM_SPAWN)

    def resetGame(self):
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = COMPRIMENTO_INICIAL
        self.prev_head = None
        self.randomFoodLocation()
        self.apple_active = False
        self.potion_active = False # <--- NOVO
        self.powerup_active = False
        self.powerup_timer = 0
        self.score = 0
        self.gameOver = False
        self.handToggleSeq = 0
        self.prevHandState = None

    def update(self, imgMain, currentHead, fingers=None):
        # Reset só se gameOver
        if self.gameOver and fingers is not None:
            if all(f == 0 for f in fingers):
                if self.prevHandState == 'open':
                    self.handToggleSeq += 1
                self.prevHandState = 'closed'
            else:
                self.prevHandState = 'open'
            if self.handToggleSeq >= 2:
                self.resetGame()

        # Tela GameOver
        if self.gameOver:
            overlay = imgMain.copy()
            cv2.rectangle(overlay, (0, 0), (imgMain.shape[1], imgMain.shape[0]), (0, 0, 0), -1)
            imgMain = cv2.addWeighted(overlay, 0.6, imgMain, 0.4, 0)
            cvzone.putTextRect(imgMain, "PERDEU!", POS_GAMEOVER_TXT, scale=SCALE_GAMEOVER_TXT, thickness=5, offset=20, colorR=(0, 0, 0), colorT=(0, 0, 255), font=cv2.FONT_HERSHEY_COMPLEX)
            
            img_pil = Image.fromarray(cv2.cvtColor(imgMain, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            text1 = "Faça o sinal de reset com a mão"
            text2 = "para jogar novamente"
            y_start = 450
            bbox1 = draw.textbbox((0, 0), text1, font=font_small)
            w1 = bbox1[2] - bbox1[0]; h1 = bbox1[3] - bbox1[1]; x1 = imgMain.shape[1] // 2 - w1 // 2
            draw.text((x1, y_start), text1, font=font_small, fill=(255, 255, 255, 255))
            bbox2 = draw.textbbox((0, 0), text2, font=font_small)
            w2 = bbox2[2] - bbox2[0]; x2 = imgMain.shape[1] // 2 - w2 // 2
            draw.text((x2, y_start + h1 + 10), text2, font=font_small, fill=(255, 255, 255, 255))
            imgMain = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # Mostra Score Final e High Score
            cvzone.putTextRect(imgMain, f'Score Final: {self.score}', POS_SCORE_TXT, scale=SCALE_SCORE_TXT, thickness=2, offset=OFFSET_TEXTO, colorR=(0, 0, 0), colorT=(0, 255, 0))
            pos_hs_gameover = [POS_SCORE_TXT[0], POS_SCORE_TXT[1] + 60]
            cvzone.putTextRect(imgMain, f'High Score: {self.high_score}', pos_hs_gameover, scale=SCALE_SCORE_TXT, thickness=2, offset=OFFSET_TEXTO, colorR=(0, 0, 0), colorT=(0, 255, 255))
            return imgMain

        if self.prev_head is None:
            self.prev_head = currentHead

        px, py = self.prev_head
        cx, cy = currentHead
        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.current_length += distance
        self.prev_head = cx, cy

        while self.current_length > self.allowed_length:
            self.current_length -= self.lengths[0]
            self.lengths.pop(0)
            self.points.pop(0)

        # Comida donut
        fx, fy = self.foodPoint
        if fx - self.wFood // 2 < cx < fx + self.wFood // 2 and fy - self.hFood // 2 < cy < fy + self.hFood // 2:
            self.randomFoodLocation()
            self.allowed_length += CRESCIMENTO_POR_COMIDA
            self.score += PONTOS_POR_COMIDA

        # Cobra verde neon
        for i in range(1, len(self.points)):
            cv2.line(imgMain, self.points[i - 1], self.points[i], COR_COBRA, GROSSURA_COBRA)
        if self.points:
            cv2.circle(imgMain, self.points[-1], GROSSURA_CABECA, COR_COBRA, cv2.FILLED)

        # Donut pulsando
        scale = 1 + 0.05 * math.sin(time.time() * 5)
        wFoodScaled = int(self.wFood * scale)
        hFoodScaled = int(self.hFood * scale)
        if wFoodScaled > 0 and hFoodScaled > 0:
            food = cv2.resize(self.imgFood, (wFoodScaled, hFoodScaled))
            imgMain = cvzone.overlayPNG(imgMain, food, (fx - wFoodScaled // 2, fy - hFoodScaled // 2))

        # Spawn maçã
        if not self.apple_active and self.score >= SCORE_PARA_MACA:
            if random.randint(0, CHANCE_MACA_SPAWN) == 1:
                self.spawnApple()
        
        # Spawn poção (NOVO)
        # Só spawna se a maçã ou outra poção não estiverem ativas
        if not self.potion_active and not self.apple_active and self.score >= SCORE_PARA_POTION:
            if random.randint(0, CHANCE_POTION_SPAWN) == 1:
                self.spawnPotion()

        # Maçã animada
        if self.apple_active and num_apple_frames > 0:
            frame = apple_frames[self.apple_frame_index]
            self.apple_frame_index = (self.apple_frame_index + 1) % num_apple_frames
            
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA)
            ax, ay = self.apple_pos
            imgMain = cvzone.overlayPNG(imgMain, frame_cv, (ax - TAMANHO_MACA // 2, ay - TAMANHO_MACA // 2))

            # Colisão maçã
            hx, hy = self.points[-1]
            if abs(hx - ax) < RAIO_COLISAO_COMIDA and abs(hy - ay) < RAIO_COLISAO_COMIDA:
                self.apple_active = False
                self.score += PONTOS_POR_MACA
                self.powerup_active = True
                self.powerup_timer = time.time()

        # Poção (NOVO)
        if self.potion_active:
            px, py = self.potion_pos
            # Desenha a poção
            imgMain = cvzone.overlayPNG(imgMain, self.imgPotion, (px - self.wPotion // 2, py - self.hPotion // 2))

            # Colisão Poção
            hx, hy = self.points[-1] # Pega a cabeça da cobra
            if abs(hx - px) < RAIO_COLISAO_COMIDA and abs(hy - py) < RAIO_COLISAO_COMIDA:
                self.potion_active = False # Poção desaparece
                
                # --- LÓGICA DE DIMINUIR A COBRA ---
                self.allowed_length = int(self.allowed_length / 2)
                
                # Garante que a cobra não fique menor que o tamanho inicial
                if self.allowed_length < COMPRIMENTO_INICIAL:
                    self.allowed_length = COMPRIMENTO_INICIAL
                # -----------------------------------

        # Power-up info
        if self.powerup_active:
            elapsed = time.time() - self.powerup_timer
            if elapsed > DURACAO_POWERUP_S:
                self.powerup_active = False
            else:
                texto_powerup = f'IMORTALIDADE ({DURACAO_POWERUP_S - int(elapsed)}s)'
                cvzone.putTextRect(imgMain, texto_powerup, POS_POWERUP_TXT, scale=SCALE_SCORE_TXT, thickness=2, offset=OFFSET_TEXTO, colorR=(0, 0, 0), colorT=(0, 255, 0))

        # Atualiza high score se necessário (durante o jogo)
        if self.score > self.high_score:
            self.high_score = self.score
            self.save_high_score()

        # Score (Durante o jogo)
        cvzone.putTextRect(imgMain, f'Score: {self.score}', POS_SCORE_TXT, scale=SCALE_SCORE_TXT, thickness=2, offset=OFFSET_TEXTO, colorR=(0, 0, 0), colorT=(0, 255, 0))
        # High Score (Durante o jogo)
        pos_hs_ingame = [LARGURA_TELA - 350, POS_SCORE_TXT[1]]
        cvzone.putTextRect(imgMain, f'High Score: {self.high_score}', pos_hs_ingame, scale=SCALE_SCORE_TXT, thickness=2, offset=OFFSET_TEXTO, colorR=(0, 0, 0), colorT=(0, 255, 255))


        # Colisão corpo (não colide se powerup ativo)
        if len(self.points) > DISTANCIA_COLISAO_CORPO and not self.powerup_active:
            hx, hy = self.points[-1]
            for pt in self.points[:-DISTANCIA_COLISAO_CORPO]:
                if math.hypot(hx - pt[0], hy - pt[1]) < RAIO_COLISAO_CORPO:
                    self.gameOver = True
                    break # Sai do loop 'for' de colisão

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
        cv2.circle(img, tuple(pontIndex), RAIO_PONTA_DEDO, COR_PONTA_DEDO, cv2.FILLED)
        img = game.update(img, pontIndex, fingers)
    else:
        # Tela inicial
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        text = "Mostre a mão para começar"
        bbox = draw.textbbox((0, 0), text, font=font_large)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        x = img.shape[1] // 2 - w // 2
        y = img.shape[0] // 2 - h // 2
        draw.rectangle([x - 10, y - 10, x + w + 10, y + h + 10], fill=(0, 0, 0, 255))
        draw.text((x, y), text, font=font_large, fill=(0, 255, 0, 255))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()