"""
=============================================================================
     SNAKE DONUTS - ARCADE EDITION (UNIMAR ABERTA - CURSO DE ADS)
=============================================================================
Recursos da versão Arcade:
- Game Juice: Sistema de partículas explosivas ao coletar itens
- Sistema de Combo & Multiplicadores dinâmicos (x1 até x5 com timer de adrenalina)
- Cobra Neon com efeito glow, gradiente e olhos dinâmicos
- Textos flutuantes animados com pontuações e mensagens
- Efeitos sonoros retrô sintetizados em background thread (sem lag de frames)
- Leaderboard do Stand (Top 5 com apelido de 3 letras e persistência em JSON)
- Suavização de rastro para movimentos macios e anti-ruído da webcam
- Câmera espelhada (efeito espelho natural para os visitantes)
- Suporte a Modo Tela Cheia (Tecla F) e Mute (Tecla M)
=============================================================================
"""

import os
import cvzone
import cv2
import numpy as np
import math
import random
import time
import json
import threading
from cvzone.HandTrackingModule import HandDetector
from PIL import ImageFont, ImageDraw, Image

# Tenta carregar winsound no Windows para efeitos sonoros arcade
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# =============================================================================
# CONFIGURAÇÕES E PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LARGURA_DESEJADA = 1280
ALTURA_DESEJADA = 720

DETECTION_CON = 0.8
MAX_HANDS = 1

# Caminhos dos Assets
ARQUIVO_MACA = os.path.join(SCRIPT_DIR, "enchanted_apple.gif")
ARQUIVO_DONUT = os.path.join(SCRIPT_DIR, "Donut.png")
ARQUIVO_POTION = os.path.join(SCRIPT_DIR, "Potion.png")
ARQUIVO_COIN = os.path.join(SCRIPT_DIR, "coin.png")
ARQUIVO_GHOST1 = os.path.join(SCRIPT_DIR, "ghost1.png")
ARQUIVO_GHOST2 = os.path.join(SCRIPT_DIR, "ghost2.png")
ARQUIVO_GHOST3 = os.path.join(SCRIPT_DIR, "ghost3.png")
ARQUIVO_LEADERBOARD = os.path.join(SCRIPT_DIR, "leaderboard.json")

# Dimensões dos Sprites
TAMANHO_DONUT = 75
TAMANHO_MACA = 75
TAMANHO_POTION = 80
TAMANHO_COIN = 85
TAMANHO_GHOST = 75

# Regras do Jogo
COMPRIMENTO_INICIAL = 160
CRESCIMENTO_POR_COMIDA = 35
MARGEM_SPAWN = 100

PONTOS_DONUT = 10
PONTOS_MACA = 25
PONTOS_COIN = 50
PONTOS_GHOST = 100

# Spawns e Durações (segundos)
DURACAO_POWERUP_S = 6.0
DURACAO_GHOST_S = 6.0
DURACAO_MACA_S = 8.0
DURACAO_POTION_S = 7.0
DURACAO_COIN_S = 6.0
TEMPO_JANELA_COMBO = 2.8   # Segundos para manter a sequência de combo
VELOCIDADE_GHOST = 4.2

# =============================================================================
# MOTOR DE SOM RETRÔ (THREAD ASSÍNCRONA)
# =============================================================================
class ArcadeAudio:
    def __init__(self):
        self.muted = False

    def play(self, sound_type):
        if not AUDIO_AVAILABLE or self.muted:
            return
        threading.Thread(target=self._play_worker, args=(sound_type,), daemon=True).start()

    def _play_worker(self, sound_type):
        try:
            if sound_type == "donut":
                winsound.Beep(650, 45)
                winsound.Beep(850, 45)
            elif sound_type == "coin":
                winsound.Beep(988, 50)
                winsound.Beep(1318, 90)
            elif sound_type == "potion":
                winsound.Beep(800, 40)
                winsound.Beep(600, 40)
                winsound.Beep(450, 60)
            elif sound_type == "powerup":
                for freq in [523, 659, 784, 1046]:
                    winsound.Beep(freq, 40)
            elif sound_type == "ghost_eat":
                winsound.Beep(400, 60)
                winsound.Beep(800, 80)
            elif sound_type == "game_over":
                for freq in [440, 370, 311, 261]:
                    winsound.Beep(freq, 90)
            elif sound_type == "combo":
                winsound.Beep(1100, 50)
                winsound.Beep(1400, 70)
        except Exception:
            pass

audio = ArcadeAudio()

# =============================================================================
# UTILITÁRIOS VISUAIS (GLASSMORPHISM CYBER-CLEAN)
# =============================================================================
def desenhar_retangulo_arredondado(img, pt1, pt2, cor_fundo, cor_borda, raio=14, alpha=0.85, espessura_borda=1):
    """Desenha card translúcido de alta performance operando direto na ROI (sem copiar a imagem inteira)."""
    x1, y1 = pt1
    x2, y2 = pt2
    ih, iw = img.shape[:2]
    x1 = max(0, min(iw - 1, x1))
    y1 = max(0, min(ih - 1, y1))
    x2 = max(0, min(iw, x2))
    y2 = max(0, min(ih, y2))
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return img

    roi = img[y1:y2, x1:x2]
    overlay_roi = np.full((h, w, 3), cor_fundo, dtype=np.uint8)
    cv2.addWeighted(overlay_roi, alpha, roi, 1.0 - alpha, 0, roi)

    if espessura_borda > 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), cor_borda, espessura_borda, cv2.LINE_AA)

    return img

# =============================================================================
# MOTOR DE PARTÍCULAS
# =============================================================================
class Particle:
    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3.0, 9.0)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.radius = random.uniform(4.0, 8.0)
        self.life = 1.0  # de 1.0 a 0.0
        self.decay = random.uniform(0.04, 0.08)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.94
        self.vy *= 0.94
        self.life -= self.decay
        self.radius = max(1.0, self.radius * 0.96)
        return self.life > 0

    def draw(self, img):
        if self.life > 0:
            alpha = max(0.1, min(1.0, self.life))
            pt = (int(self.x), int(self.y))
            cv2.circle(img, pt, int(self.radius), self.color, -1)

# =============================================================================
# TEXTO FLUTUANTE (FLOATING TEXT)
# =============================================================================
class FloatingText:
    def __init__(self, text, x, y, color=(0, 255, 255), scale=1.0):
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.scale = scale
        self.life = 1.0
        self.decay = 0.035
        self.vy = -2.2

    def update(self):
        self.y += self.vy
        self.life -= self.decay
        return self.life > 0

    def draw(self, img):
        if self.life > 0:
            cv2.putText(
                img, self.text, (int(self.x), int(self.y)),
                cv2.FONT_HERSHEY_DUPLEX, self.scale, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                img, self.text, (int(self.x), int(self.y)),
                cv2.FONT_HERSHEY_DUPLEX, self.scale, self.color, 2, cv2.LINE_AA
            )

# =============================================================================
# GERENCIADOR DO LEADERBOARD (STAND TOP 5)
# =============================================================================
class StandLeaderboard:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scores = self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return sorted(data, key=lambda x: x['score'], reverse=True)[:5]
            except Exception:
                pass
        # Padrão inicial com referências de ADS Unimar
        default_scores = [
            {"name": "ADS", "score": 350, "date": "Recorde"},
            {"name": "UNI", "score": 240, "date": "Stand"},
            {"name": "DEV", "score": 180, "date": "Top"},
            {"name": "SNK", "score": 120, "date": "Arcade"},
            {"name": "BOB", "score": 80,  "date": "Player"},
        ]
        self.save(default_scores)
        return default_scores

    def save(self, data=None):
        if data is None:
            data = self.scores
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar leaderboard: {e}")

    def is_top_score(self, score):
        if len(self.scores) < 5:
            return True
        return score > self.scores[-1]['score']

    def add_score(self, name, score):
        entry = {
            "name": name.upper()[:3],
            "score": score,
            "date": time.strftime("%H:%M")
        }
        self.scores.append(entry)
        self.scores = sorted(self.scores, key=lambda x: x['score'], reverse=True)[:5]
        self.save()

    def get_high_score(self):
        if self.scores:
            return self.scores[0]['score']
        return 0

    def update_high_score(self, score):
        current_high = self.get_high_score()
        if score > current_high:
            self.scores.insert(0, {
                "name": "RECORDE",
                "score": score,
                "date": time.strftime("%H:%M")
            })
            self.scores = sorted(self.scores, key=lambda x: x['score'], reverse=True)[:5]
            self.save()
            return True
        return False

# =============================================================================
# CARREGAMENTO OTIMIZADO DA MAÇÃ ENCANTADA (GIF)
# =============================================================================
def carregar_frames_maca():
    frames = []
    if os.path.exists(ARQUIVO_MACA):
        try:
            gif = Image.open(ARQUIVO_MACA)
            # Amostra a cada 4 frames para carregamento instantâneo
            step = 4
            for frame_idx in range(0, getattr(gif, 'n_frames', 1), step):
                gif.seek(frame_idx)
                f_rgba = gif.convert("RGBA").resize((TAMANHO_MACA, TAMANHO_MACA))
                frames.append(cv2.cvtColor(np.array(f_rgba), cv2.COLOR_RGBA2BGRA))
        except Exception as e:
            print(f"Aviso ao carregar GIF da maçã: {e}")
    return frames

# =============================================================================
# CLASSE PRINCIPAL DO JOGO
# =============================================================================
class SnakeArcade:
    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura
        self.leaderboard = StandLeaderboard(ARQUIVO_LEADERBOARD)

        # Rastro e Cobra
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = COMPRIMENTO_INICIAL
        self.smooth_head = None

        # Carregar Sprites
        self.imgDonut = self._load_sprite(ARQUIVO_DONUT, TAMANHO_DONUT, (255, 105, 180))
        self.imgPotion = self._load_sprite(ARQUIVO_POTION, TAMANHO_POTION, (255, 0, 0))
        self.imgCoin = self._load_sprite(ARQUIVO_COIN, TAMANHO_COIN, (0, 215, 255))
        self.imgGhost1 = self._load_sprite(ARQUIVO_GHOST1, TAMANHO_GHOST, (180, 180, 180))
        self.imgGhost2 = self._load_sprite(ARQUIVO_GHOST2, TAMANHO_GHOST, (180, 180, 180))
        self.imgGhost3 = self._load_sprite(ARQUIVO_GHOST3, TAMANHO_GHOST, (255, 50, 50))
        self.apple_frames = carregar_frames_maca()
        self.apple_frame_idx = 0

        # Partículas e Efeitos
        self.particles = []
        self.floating_texts = []
        self.shake_time = 0

        # Estados de Itens
        self.food_pos = (0, 0)
        self.spawn_food()

        self.apple_active = False
        self.apple_pos = (0, 0)
        self.apple_timer = 0

        self.potion_active = False
        self.potion_pos = (0, 0)
        self.potion_timer = 0

        self.coin_active = False
        self.coin_pos = (0, 0)
        self.coin_timer = 0

        self.ghost_active = False
        self.ghost_pos = [0.0, 0.0]
        self.ghost_timer = 0
        self.ghost_dir = 1

        self.powerup_active = False
        self.powerup_timer = 0

        # Sistema de Combo
        self.combo_count = 0
        self.combo_multiplier = 1
        self.last_eat_time = 0
        self.max_combo_reached = 1

        # Pontuação e Ciclo de Vida
        self.score = 0
        self.game_over = False
        self.score_scale_anim = 1.0

        # Gestos e Reset
        self.hand_closed_counter = 0
        self.last_hand_state = None

        # Nome para o Ranking em caso de novo recorde
        self.player_initials = ["A", "D", "S"]
        self.selected_initial_idx = 0
        self.record_registered = False

    def _load_sprite(self, path, size, fallback_color):
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return cv2.resize(img, (size, size))
        # Fallback gerado proceduralmente
        img = np.zeros((size, size, 4), dtype=np.uint8)
        cv2.circle(img, (size // 2, size // 2), size // 2 - 2, fallback_color + (255,), -1)
        return img

    def spawn_food(self):
        self.food_pos = (
            random.randint(MARGEM_SPAWN, self.largura - MARGEM_SPAWN),
            random.randint(MARGEM_SPAWN + 40, self.altura - MARGEM_SPAWN)
        )

    def spawn_apple(self):
        self.apple_active = True
        self.apple_timer = time.time()
        self.apple_pos = (
            random.randint(MARGEM_SPAWN, self.largura - MARGEM_SPAWN),
            random.randint(MARGEM_SPAWN + 40, self.altura - MARGEM_SPAWN)
        )

    def spawn_potion(self):
        self.potion_active = True
        self.potion_timer = time.time()
        self.potion_pos = (
            random.randint(MARGEM_SPAWN, self.largura - MARGEM_SPAWN),
            random.randint(MARGEM_SPAWN + 40, self.altura - MARGEM_SPAWN)
        )

    def spawn_coin(self):
        self.coin_active = True
        self.coin_timer = time.time()
        self.coin_pos = (
            random.randint(MARGEM_SPAWN, self.largura - MARGEM_SPAWN),
            random.randint(MARGEM_SPAWN + 40, self.altura - MARGEM_SPAWN)
        )

    def spawn_ghost(self):
        self.ghost_active = True
        self.ghost_timer = time.time()
        # Spawna afastado da cabeça da cobra
        hx, hy = self.points[-1] if self.points else (self.largura // 2, self.altura // 2)
        gx = MARGEM_SPAWN if hx > self.largura // 2 else self.largura - MARGEM_SPAWN
        gy = random.randint(MARGEM_SPAWN + 40, self.altura - MARGEM_SPAWN)
        self.ghost_pos = [float(gx), float(gy)]

    def emit_particles(self, x, y, color, count=16):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def trigger_shake(self, duration=0.3):
        self.shake_time = time.time() + duration

    def reset_game(self):
        self.points.clear()
        self.lengths.clear()
        self.current_length = 0
        self.allowed_length = COMPRIMENTO_INICIAL
        self.smooth_head = None
        self.spawn_food()

        self.apple_active = False
        self.potion_active = False
        self.coin_active = False
        self.ghost_active = False
        self.powerup_active = False

        self.particles.clear()
        self.floating_texts.clear()

        self.score = 0
        self.combo_count = 0
        self.combo_multiplier = 1
        self.max_combo_reached = 1
        self.game_over = False
        self.record_registered = False
        self.hand_closed_counter = 0

    def register_eat(self, base_points, item_name, pos, color):
        now = time.time()
        if now - self.last_eat_time <= TEMPO_JANELA_COMBO:
            self.combo_count += 1
            self.combo_multiplier = min(5, 1 + self.combo_count // 2)
            if self.combo_multiplier > self.max_combo_reached:
                self.max_combo_reached = self.combo_multiplier
                self.floating_texts.append(
                    FloatingText(f"COMBO x{self.combo_multiplier}!", pos[0], pos[1] - 40, (0, 255, 255), 1.1)
                )
                audio.play("combo")
        else:
            self.combo_count = 1
            self.combo_multiplier = 1

        self.last_eat_time = now
        gained = base_points * self.combo_multiplier
        self.score += gained
        self.score_scale_anim = 1.4

        # Texto Flutuante
        mult_str = f" (x{self.combo_multiplier})" if self.combo_multiplier > 1 else ""
        self.floating_texts.append(
            FloatingText(f"+{gained}{mult_str}", pos[0], pos[1], color, 0.9)
        )
        self.emit_particles(pos[0], pos[1], color, count=18)

    # -------------------------------------------------------------------------
    # ATUALIZAÇÃO PRINCIPAL DO FRAME
    # -------------------------------------------------------------------------
    def update(self, imgMain, raw_head, fingers=None):
        now = time.time()

        # Verifica decaimento de combo
        if self.combo_multiplier > 1 and (now - self.last_eat_time > TEMPO_JANELA_COMBO):
            self.combo_count = 0
            self.combo_multiplier = 1

        # Gesto de Reset com Mão Fechada
        if fingers is not None:
            is_closed = all(f == 0 for f in fingers)
            if self.last_hand_state != is_closed:
                if is_closed and self.game_over:
                    self.hand_closed_counter += 1
                    if self.hand_closed_counter >= 2:
                        self.reset_game()
                self.last_hand_state = is_closed

        # TELA DE GAME OVER
        if self.game_over:
            return self._render_game_over(imgMain)

        # Suavização da ponta do dedo (Filtro Passa-Baixa)
        rx, ry = raw_head
        if self.smooth_head is None:
            self.smooth_head = [float(rx), float(ry)]
        else:
            alpha_smooth = 0.55
            self.smooth_head[0] = self.smooth_head[0] * (1 - alpha_smooth) + rx * alpha_smooth
            self.smooth_head[1] = self.smooth_head[1] * (1 - alpha_smooth) + ry * alpha_smooth

        cx, cy = int(self.smooth_head[0]), int(self.smooth_head[1])

        # Adiciona pontos à cobra
        if self.points:
            px, py = self.points[-1]
            dist = math.hypot(cx - px, cy - py)
            if dist > 3:  # Amortecimento de microestagnação
                self.points.append([cx, cy])
                self.lengths.append(dist)
                self.current_length += dist
        else:
            self.points.append([cx, cy])
            self.lengths.append(0)

        # Limita o comprimento
        while self.current_length > self.allowed_length and len(self.lengths) > 1:
            self.current_length -= self.lengths.pop(0)
            self.points.pop(0)

        # ---------------------------------------------------------------------
        # SPAWN DE ITENS ESPECIAIS
        # ---------------------------------------------------------------------
        if not self.apple_active and not self.potion_active and not self.coin_active:
            dice = random.randint(1, 100)
            if self.score >= 50 and dice <= 2:
                self.spawn_apple()
            elif self.score >= 70 and dice <= 3:
                self.spawn_potion()
            elif self.score >= 30 and dice <= 4:
                self.spawn_coin()

        # ---------------------------------------------------------------------
        # COLISÃO COM O DONUT
        # ---------------------------------------------------------------------
        fx, fy = self.food_pos
        if math.hypot(cx - fx, cy - fy) < (TAMANHO_DONUT // 2 + 15):
            self.register_eat(PONTOS_DONUT, "DONUT", (fx, fy), (255, 105, 180))
            self.allowed_length += CRESCIMENTO_POR_COMIDA
            self.spawn_food()
            audio.play("donut")

        # ---------------------------------------------------------------------
        # COLISÃO COM MAÇÃ ENCANTADA (POWER-UP)
        # ---------------------------------------------------------------------
        if self.apple_active:
            if now - self.apple_timer > DURACAO_MACA_S:
                self.apple_active = False
            else:
                ax, ay = self.apple_pos
                if math.hypot(cx - ax, cy - ay) < (TAMANHO_MACA // 2 + 15):
                    self.apple_active = False
                    self.powerup_active = True
                    self.powerup_timer = now
                    self.register_eat(PONTOS_MACA, "POWER-UP!", (ax, ay), (255, 215, 0))
                    audio.play("powerup")
                    self.floating_texts.append(
                        FloatingText("INVENCÍVEL!", ax, ay - 30, (0, 255, 255), 1.2)
                    )

        # ---------------------------------------------------------------------
        # COLISÃO COM POÇÃO (ENCURTAR COBRA)
        # ---------------------------------------------------------------------
        if self.potion_active:
            if now - self.potion_timer > DURACAO_POTION_S:
                self.potion_active = False
            else:
                px_pos, py_pos = self.potion_pos
                if math.hypot(cx - px_pos, cy - py_pos) < (TAMANHO_POTION // 2 + 15):
                    self.potion_active = False
                    self.allowed_length = max(COMPRIMENTO_INICIAL, int(self.allowed_length * 0.55))
                    self.register_eat(15, "ENCURTOU!", (px_pos, py_pos), (255, 200, 0))
                    audio.play("potion")
                    self.floating_texts.append(
                        FloatingText("TAMANHO REDUZIDO!", px_pos, py_pos - 30, (255, 180, 0), 1.0)
                    )

        # ---------------------------------------------------------------------
        # COLISÃO COM MOEDA (PONTOS + SPAWN FANTASMA)
        # ---------------------------------------------------------------------
        if self.coin_active:
            if now - self.coin_timer > DURACAO_COIN_S:
                self.coin_active = False
            else:
                coin_x, coin_y = self.coin_pos
                if math.hypot(cx - coin_x, cy - coin_y) < (TAMANHO_COIN // 2 + 15):
                    self.coin_active = False
                    self.register_eat(PONTOS_COIN, "SUPER MOEDA!", (coin_x, coin_y), (0, 220, 255))
                    audio.play("coin")
                    self.spawn_ghost()
                    self.floating_texts.append(
                        FloatingText("FANTASMA LIBERADO!", coin_x, coin_y - 30, (50, 50, 255), 1.0)
                    )

        # ---------------------------------------------------------------------
        # LÓGICA DO FANTASMA
        # ---------------------------------------------------------------------
        if self.ghost_active:
            if now - self.ghost_timer > DURACAO_GHOST_S:
                self.ghost_active = False
            else:
                gx, gy = self.ghost_pos
                dx = cx - gx
                dy = cy - gy
                dist_g = math.hypot(dx, dy)
                if dist_g > 1:
                    ndx = (dx / dist_g) * VELOCIDADE_GHOST
                    ndy = (dy / dist_g) * VELOCIDADE_GHOST

                    if self.powerup_active:
                        # Fantasma FOGE da cobra!
                        ndx = -ndx
                        ndy = -ndy

                    self.ghost_dir = 1 if ndx > 0 else -1
                    gx = max(MARGEM_SPAWN, min(self.largura - MARGEM_SPAWN, gx + ndx))
                    gy = max(MARGEM_SPAWN + 40, min(self.altura - MARGEM_SPAWN, gy + ndy))
                    self.ghost_pos = [gx, gy]

                # Colisão Fantasma x Cabeça
                if math.hypot(cx - gx, cy - gy) < (TAMANHO_GHOST // 2 + 12):
                    if self.powerup_active:
                        self.ghost_active = False
                        self.register_eat(PONTOS_GHOST, "FANTASMA COMIDO!", (int(gx), int(gy)), (0, 255, 255))
                        audio.play("ghost_eat")
                    else:
                        self._trigger_game_over()
                        return self._render_game_over(imgMain)

        # ---------------------------------------------------------------------
        # AUTO-COLISÃO DA COBRA (CORPO)
        # ---------------------------------------------------------------------
        if len(self.points) > 25 and not self.powerup_active:
            hx, hy = self.points[-1]
            # Verifica pontos anteriores deixando margem de segurança no pescoço
            for pt in self.points[:-22]:
                if math.hypot(hx - pt[0], hy - pt[1]) < 18:
                    self._trigger_game_over()
                    return self._render_game_over(imgMain)

        # ---------------------------------------------------------------------
        # RENDERIZAÇÃO NA TELA
        # ---------------------------------------------------------------------
        # Efeito Screen Shake se ativo
        if now < self.shake_time:
            ox = random.randint(-4, 4)
            oy = random.randint(-4, 4)
            M = np.float32([[1, 0, ox], [0, 1, oy]])
            imgMain = cv2.warpAffine(imgMain, M, (self.largura, self.altura))

        # 1. Desenha a Cobra Neon com Gradiente
        self._render_snake(imgMain)

        # 2. Desenha Donut Pulsante
        scale_donut = 1.0 + 0.08 * math.sin(now * 6)
        w_d = int(TAMANHO_DONUT * scale_donut)
        h_d = int(TAMANHO_DONUT * scale_donut)
        donut_resized = cv2.resize(self.imgDonut, (w_d, h_d))
        imgMain = cvzone.overlayPNG(imgMain, donut_resized, (fx - w_d // 2, fy - h_d // 2))

        # 3. Desenha Maçã Encantada
        if self.apple_active and self.apple_frames:
            ax, ay = self.apple_pos
            frame_maca = self.apple_frames[self.apple_frame_idx % len(self.apple_frames)]
            self.apple_frame_idx += 1
            imgMain = cvzone.overlayPNG(imgMain, frame_maca, (ax - TAMANHO_MACA // 2, ay - TAMANHO_MACA // 2))

        # 4. Desenha Poção
        if self.potion_active:
            px_pos, py_pos = self.potion_pos
            imgMain = cvzone.overlayPNG(imgMain, self.imgPotion, (px_pos - TAMANHO_POTION // 2, py_pos - TAMANHO_POTION // 2))

        # 5. Desenha Moeda
        if self.coin_active:
            coin_x, coin_y = self.coin_pos
            # Moeda com brilho pulsante
            scale_c = 1.0 + 0.06 * math.sin(now * 8)
            wc = int(TAMANHO_COIN * scale_c)
            hc = int(TAMANHO_COIN * scale_c)
            coin_resized = cv2.resize(self.imgCoin, (wc, hc))
            imgMain = cvzone.overlayPNG(imgMain, coin_resized, (coin_x - wc // 2, coin_y - hc // 2))

        # 6. Desenha Fantasma
        if self.ghost_active:
            gx, gy = int(self.ghost_pos[0]), int(self.ghost_pos[1])
            if self.powerup_active:
                sprite_g = self.imgGhost3
            else:
                sprite_g = self.imgGhost2 if self.ghost_dir == 1 else self.imgGhost1

            # Pisca nos últimos 1.5s
            tempo_restante = DURACAO_GHOST_S - (now - self.ghost_timer)
            piscar = tempo_restante < 1.8 and int(now * 8) % 2 == 0
            if not piscar:
                imgMain = cvzone.overlayPNG(imgMain, sprite_g, (gx - TAMANHO_GHOST // 2, gy - TAMANHO_GHOST // 2))

        # 7. Atualiza e Desenha Partículas
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles:
            p.draw(imgMain)

        # 8. Atualiza e Desenha Textos Flutuantes
        self.floating_texts = [t for t in self.floating_texts if t.update()]
        for t in self.floating_texts:
            t.draw(imgMain)

        # 9. Desenha o HUD Arcade no Topo
        self._render_hud(imgMain, now)

        return imgMain

    # -------------------------------------------------------------------------
    # DESENHO DA COBRA NEON COM GLOW E OLHOS
    # -------------------------------------------------------------------------
    def _render_snake(self, img):
        n_pts = len(self.points)
        if n_pts < 2:
            return

        is_power = self.powerup_active
        # Desenha segmentos com gradiente e largura variável
        for i in range(1, n_pts):
            p1 = tuple(self.points[i - 1])
            p2 = tuple(self.points[i])

            # Fator de 0.0 (cauda) a 1.0 (cabeça)
            factor = i / n_pts
            thickness = int(10 + factor * 14)

            if is_power:
                # Efeito Mágico Arco-íris / Dourado
                hue = int((time.time() * 90 + factor * 180) % 180)
                hsv_pixel = np.uint8([[[hue, 255, 255]]])
                bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
                color = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
            else:
                # Gradiente Neon: Ciano no rastro -> Verde Neon na cabeça
                b = int(255 * (1 - factor))
                g = 255
                r = int(50 * factor)
                color = (b, g, r)

            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

        # Glow na cabeça
        hx, hy = self.points[-1]
        glow_radius = 28 if not is_power else 34
        glow_color = (0, 255, 255) if is_power else (0, 255, 120)

        overlay = img.copy()
        cv2.circle(overlay, (hx, hy), glow_radius, glow_color, -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.circle(img, (hx, hy), 16, (255, 255, 255), -1, cv2.LINE_AA)

        # Olhos expressivos direcionados para o Donut
        fx, fy = self.food_pos
        ang = math.atan2(fy - hy, fx - hx)
        eye_dist = 6
        eye1_x = int(hx + math.cos(ang - 0.5) * eye_dist)
        eye1_y = int(hy + math.sin(ang - 0.5) * eye_dist)
        eye2_x = int(hx + math.cos(ang + 0.5) * eye_dist)
        eye2_y = int(hy + math.sin(ang + 0.5) * eye_dist)

        cv2.circle(img, (eye1_x, eye1_y), 3, (0, 0, 0), -1)
        cv2.circle(img, (eye2_x, eye2_y), 3, (0, 0, 0), -1)

    # -------------------------------------------------------------------------
    # HUD ARCADE RETRÔ (PAINEL SUPERIOR)
    # -------------------------------------------------------------------------
    def _render_hud(self, img, now):
        # Barra superior translúcida estilo Cyber-Clean (operando via ROI)
        hud_h = 62
        hud_roi = img[0:hud_h, 0:self.largura]
        hud_bg = np.full(hud_roi.shape, (12, 14, 22), dtype=np.uint8)
        cv2.addWeighted(hud_bg, 0.82, hud_roi, 0.18, 0, hud_roi)
        cv2.line(img, (0, hud_h), (self.largura, hud_h), (0, 220, 255), 2, cv2.LINE_AA)

        # Tag do Evento
        cv2.putText(
            img, "ADS * UNIMAR ABERTA", (24, 25),
            cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            img, "SNAKE DONUTS ARCADE", (24, 50),
            cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Score com animação e cor Neon
        self.score_scale_anim = max(1.0, self.score_scale_anim - 0.03)
        score_text = f"SCORE: {self.score}"
        cv2.putText(
            img, score_text, (self.largura // 2 - 120, 42),
            cv2.FONT_HERSHEY_DUPLEX, 1.05 * self.score_scale_anim, (0, 255, 140), 2, cv2.LINE_AA
        )

        # High Score
        high_score = max(self.score, self.leaderboard.get_high_score())
        hs_text = f"RECORDE: {high_score}"
        cv2.putText(
            img, hs_text, (self.largura - 260, 42),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA
        )

        # Indicador de Multiplicador de Combo
        if self.combo_multiplier > 1:
            tempo_combo_restante = max(0.0, TEMPO_JANELA_COMBO - (now - self.last_eat_time))
            progresso_combo = tempo_combo_restante / TEMPO_JANELA_COMBO

            # Caixa de combo neon
            bx, by, bw, bh = self.largura // 2 - 110, 72, 220, 28
            desenhar_retangulo_arredondado(img, (bx, by), (bx + bw, by + bh), (10, 14, 26), (0, 220, 255), raio=8, alpha=0.90)

            # Barra decrescente
            w_prog = int(bw * progresso_combo)
            if w_prog > 0:
                cv2.rectangle(img, (bx + 2, by + 2), (bx + w_prog - 2, by + bh - 2), (0, 160, 255), -1)

            combo_lbl = f"COMBO x{self.combo_multiplier}!"
            cv2.putText(
                img, combo_lbl, (bx + 35, by + 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
            )

        # Barra de Power-up se ativo
        if self.powerup_active:
            elapsed_p = now - self.powerup_timer
            if elapsed_p > DURACAO_POWERUP_S:
                self.powerup_active = False
            else:
                rem_p = DURACAO_POWERUP_S - elapsed_p
                prog_p = rem_p / DURACAO_POWERUP_S
                px_b = 20
                py_b = self.altura - 75
                pw_b = 250
                ph_b = 26

                desenhar_retangulo_arredondado(img, (px_b, py_b), (px_b + pw_b, py_b + ph_b), (10, 14, 24), (0, 255, 255), raio=8, alpha=0.88)
                w_fill = int(pw_b * prog_p)
                if w_fill > 0:
                    cv2.rectangle(img, (px_b + 2, py_b + 2), (px_b + w_fill - 2, py_b + ph_b - 2), (0, 220, 255), -1)
                cv2.putText(
                    img, f"INVENCIVEL: {rem_p:.1f}s", (px_b + 12, py_b + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
                )

    def _trigger_game_over(self):
        self.game_over = True
        self.trigger_shake(0.5)
        audio.play("game_over")
        self.hand_closed_counter = 0
        self.is_new_record = self.leaderboard.update_high_score(self.score)
        if self.is_new_record:
            audio.play("powerup")

    # -------------------------------------------------------------------------
    # TELA DE GAME OVER LIMPA & IMPACTANTE (SCORE VS RECORDE)
    # -------------------------------------------------------------------------
    def _render_game_over(self, img):
        cx = self.largura // 2
        cy = self.altura // 2

        # Card Central Glassmorphism
        card_w, card_h = 580, 340
        card_x = cx - card_w // 2
        card_y = cy - card_h // 2 - 20

        desenhar_retangulo_arredondado(
            img, (card_x, card_y), (card_x + card_w, card_y + card_h),
            cor_fundo=(10, 12, 20), cor_borda=(0, 220, 255), raio=16, alpha=0.92, espessura_borda=2
        )

        # Título GAME OVER (Neon Red & White)
        cv2.putText(
            img, "GAME OVER", (cx - 175, card_y + 60),
            cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 255), 5, cv2.LINE_AA
        )
        cv2.putText(
            img, "GAME OVER", (cx - 175, card_y + 60),
            cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.line(img, (card_x + 30, card_y + 80), (card_x + card_w - 30, card_y + 80), (45, 50, 70), 1, cv2.LINE_AA)

        # SEU SCORE (Destaque Principal)
        cv2.putText(
            img, "SUA PONTUACAO", (cx - 85, card_y + 115),
            cv2.FONT_HERSHEY_DUPLEX, 0.52, (170, 175, 190), 1, cv2.LINE_AA
        )
        score_str = f"{self.score} PONTOS"
        text_size = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)[0]
        cv2.putText(
            img, score_str, (cx - text_size[0] // 2, card_y + 170),
            cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 140), 3, cv2.LINE_AA
        )

        # MELHOR RECORDE DO EVENTO
        high_score = self.leaderboard.get_high_score()
        hs_box_y = card_y + 195
        hs_box_w = 460
        hs_box_x = cx - hs_box_w // 2

        borda_hs = (0, 255, 140) if getattr(self, 'is_new_record', False) else (0, 215, 255)
        desenhar_retangulo_arredondado(
            img, (hs_box_x, hs_box_y), (hs_box_x + hs_box_w, hs_box_y + 55),
            cor_fundo=(16, 20, 32), cor_borda=borda_hs,
            raio=10, alpha=0.90, espessura_borda=1
        )

        if getattr(self, 'is_new_record', False):
            cv2.putText(
                img, "PARABENS! NOVO RECORDE DO STAND!", (hs_box_x + 25, hs_box_y + 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 140), 2, cv2.LINE_AA
            )
        else:
            hs_lbl = f"MELHOR RECORDE DO EVENTO: {high_score} PTS"
            txt_hs_size = cv2.getTextSize(hs_lbl, cv2.FONT_HERSHEY_DUPLEX, 0.58, 2)[0]
            cv2.putText(
                img, hs_lbl, (cx - txt_hs_size[0] // 2, hs_box_y + 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 215, 255), 2, cv2.LINE_AA
            )

        # Instruções de reinício rápidas e sem atrito
        cv2.putText(
            img, "Pressione [R] ou [ESPACO] para jogar de novo", (cx - 200, card_y + 285),
            cv2.FONT_HERSHEY_DUPLEX, 0.54, (220, 225, 235), 1, cv2.LINE_AA
        )
        cv2.putText(
            img, "(Ou feche a mao 2 vezes na camera)", (cx - 150, card_y + 315),
            cv2.FONT_HERSHEY_DUPLEX, 0.44, (140, 145, 160), 1, cv2.LINE_AA
        )

        return img

    def handle_key_input(self, key):
        if not self.game_over:
            return

        # Reiniciar jogo com tecla R ou Espaço
        if key in [ord('r'), ord('R'), 32]:
            self.reset_game()

# =============================================================================
# BUSCA INTELIGENTE DE CÂMERA
# =============================================================================
def encontrar_camera():
    for idx in [1, 0, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[OK] Câmera conectada com sucesso no índice: {idx}")
                return cap
            cap.release()
    return None

# =============================================================================
# LAUNCHER / EXECUÇÃO DO JOGO
# =============================================================================
def main():
    print("Iniciando Snake Donuts - Arcade Edition (ADS Unimar)...")
    cap = encontrar_camera()
    if cap is None:
        print("[ERRO] Nenhuma webcam detectada!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_DESEJADA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_DESEJADA)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, test_frame = cap.read()
    if ret:
        altura_real, largura_real, _ = test_frame.shape
    else:
        altura_real, largura_real = ALTURA_DESEJADA, LARGURA_DESEJADA

    detector = HandDetector(detectionCon=DETECTION_CON, maxHands=MAX_HANDS)
    game = SnakeArcade(largura_real, altura_real)

    nome_janela = "Snake Donuts - Arcade Edition | ADS UNIMAR ABERTA"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    fullscreen = False
    espelhar_video = True

    fps_tempo = time.time()
    fps_cont = 0
    fps_display = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Espelhamento horizontal (comutável via tecla ESPAÇO)
        if espelhar_video:
            img = cv2.flip(img, 1)

        # OTIMIZAÇÃO CRÍTICA DE FPS (Reduz inferência do MediaPipe de 45ms para 10ms):
        scale_w, scale_h = 640, 360
        img_small = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
        hands = detector.findHands(img_small, draw=False, flipType=False)

        fingers = None
        raw_head = None

        if hands:
            scale_factor_x = largura_real / scale_w
            scale_factor_y = altura_real / scale_h
            lmList = hands[0]['lmList']
            fingers = detector.fingersUp(hands[0])
            # Ponto 8 = Ponta do dedo indicador mapeada para a resolução da tela
            raw_head = (int(lmList[8][0] * scale_factor_x), int(lmList[8][1] * scale_factor_y))
            img = game.update(img, raw_head, fingers)
        else:
            if not game.game_over:
                # Alerta visual para o visitante posicionar a mão
                cx_box = largura_real // 2 - 250
                cy_box = altura_real // 2 - 40
                desenhar_retangulo_arredondado(
                    img, (cx_box, cy_box), (cx_box + 500, cy_box + 80),
                    cor_fundo=(10, 12, 20), cor_borda=(0, 220, 255), raio=14, alpha=0.88, espessura_borda=2
                )
                cv2.putText(
                    img, "MOSTRE SUA MAO PARA JOGAR!",
                    (cx_box + 40, cy_box + 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.80, (0, 255, 140), 2, cv2.LINE_AA
                )
            else:
                img = game.update(img, (0, 0), None)

        # ---------------------------------------------------------------------
        # RODAPÉ COM ATALHOS E INDICAÇÃO DE STATUS (ROI sem cópia de frame)
        # ---------------------------------------------------------------------
        foot_h = 38
        foot_roi = img[altura_real - foot_h:altura_real, 0:largura_real]
        foot_bg = np.full(foot_roi.shape, (10, 12, 18), dtype=np.uint8)
        cv2.addWeighted(foot_bg, 0.88, foot_roi, 0.12, 0, foot_roi)

        status_espelho = "LIGADO" if espelhar_video else "DESLIGADO"
        cor_espelho = (0, 255, 140) if espelhar_video else (0, 220, 255)
        status_audio = "LIGADO" if not audio.muted else "MUDO"
        cor_audio = (0, 255, 140) if not audio.muted else (0, 100, 255)

        cv2.putText(
            img, "ADS * UNIMAR ABERTA | COBRINHA ARCADE", (20, altura_real - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 140), 1, cv2.LINE_AA
        )

        cv2.putText(
            img, f"[ESPACO] Espelho: {status_espelho}", (360, altura_real - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.44, cor_espelho, 1, cv2.LINE_AA
        )

        cv2.putText(
            img, f"[M] Som: {status_audio}", (610, altura_real - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.44, cor_audio, 1, cv2.LINE_AA
        )

        cv2.putText(
            img, "[R: Reiniciar] | [TAB: Tela Cheia] | [ESC: Sair]",
            (largura_real // 2 + 130, altura_real - 14), cv2.FONT_HERSHEY_DUPLEX, 0.44, (180, 185, 200), 1, cv2.LINE_AA
        )

        # Cálculo de FPS
        fps_cont += 1
        if time.time() - fps_tempo >= 1.0:
            fps_display = fps_cont
            fps_cont = 0
            fps_tempo = time.time()

        cv2.putText(
            img, f"{fps_display} FPS", (largura_real - 85, altura_real - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 140), 1, cv2.LINE_AA
        )

        cv2.imshow(nome_janela, img)
        key = cv2.waitKey(1) & 0xFF

        # Tratamento de Teclas
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC ou Q para sair
            break
        elif key == 9 or key == ord('f') or key == ord('F'):  # TAB ou F para Tela Cheia
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == 32:  # ESPAÇO: Espelho durante jogo, ou reset/salvar recorde no game over
            if not game.game_over:
                espelhar_video = not espelhar_video
            else:
                game.handle_key_input(key)
        elif key == ord('r') or key == ord('R'):  # R: Reiniciar partida
            if game.game_over:
                game.handle_key_input(key)
            else:
                game.reset_game()
        elif key == ord('m') or key == ord('M'):  # M: Mute / Som
            audio.muted = not audio.muted
            print(f"[AUDIO] Som {'MUTADO' if audio.muted else 'ATIVADO'}")
        else:
            game.handle_key_input(key)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
