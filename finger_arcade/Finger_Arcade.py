"""
=============================================================================
     FINGER ARCADE & GESTURE ARENA - ADS UNIMAR ABERTA (EDIÇÃO STAND)
=============================================================================
3 Modos Interativos em 1 Único Projeto:
  [F1] MODO 1: DASHBOARD BIOMÉTRICO SCI-FI (HUD Holográfico 0 a 10 Dedos)
  [F2] MODO 2: MATH BLITZ & REAÇÃO RÁPIDA (Desafio Contra o Relógio de 45s)
  [F3] MODO 3: JOKENPÔ ARCADE (Pedra, Papel e Tesoura contra o Robô de ADS)

Recursos Técnicos:
- Rastreamento simultâneo de até 2 Mãos com MediaPipe (0 a 10 dedos com alta precisão).
- Reconhecimento automático de gestos (Paz & Amor, Rock, Hang Loose, Joinha, etc.).
- Performance de 60 FPS com inferência em resolução otimizada (640x360).
- Efeitos sonoros procedurais via winsound em threads assíncronas.
- Visual Cyber-Clean Glassmorphism padronizado com os outros jogos do stand.
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import math
import random
import json
import threading

try:
    import winsound
    def tocar_som_acerto():
        winsound.Beep(988, 70)
        winsound.Beep(1318, 110)
    def tocar_som_erro():
        winsound.Beep(330, 140)
    def tocar_som_tick():
        winsound.Beep(880, 40)
    def tocar_som_vitoria():
        threading.Thread(target=_worker_vitoria, daemon=True).start()
    def _worker_vitoria():
        for freq in [523, 659, 784, 1046]:
            winsound.Beep(freq, 60)
    def tocar_som_derrota():
        threading.Thread(target=_worker_derrota, daemon=True).start()
    def _worker_derrota():
        for freq in [440, 370, 311]:
            winsound.Beep(freq, 90)
    def tocar_som_empate():
        winsound.Beep(600, 80)
except Exception:
    def tocar_som_acerto(): pass
    def tocar_som_erro(): pass
    def tocar_som_tick(): pass
    def tocar_som_vitoria(): pass
    def tocar_som_derrota(): pass
    def tocar_som_empate(): pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_RECORDE = os.path.join(SCRIPT_DIR, "recorde_math.json")

def carregar_recorde():
    if os.path.exists(FILE_RECORDE):
        try:
            with open(FILE_RECORDE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("recorde", 0)
        except Exception:
            return 0
    return 0

def salvar_recorde(novo_score):
    try:
        with open(FILE_RECORDE, "w", encoding="utf-8") as f:
            json.dump({"recorde": novo_score, "atualizado": time.strftime("%d/%m/%Y %H:%M")}, f, indent=2)
    except Exception as e:
        print(f"[ERRO RECORDE] {e}")

# =============================================================================
# SISTEMA DE PARTÍCULAS / CONFETES
# =============================================================================
class Particula:
    def __init__(self, x, y, cor=None):
        self.x = float(x)
        self.y = float(y)
        ang = random.uniform(0, 2 * math.pi)
        vel = random.uniform(4.0, 12.0)
        self.vx = math.cos(ang) * vel
        self.vy = math.sin(ang) * vel - random.uniform(2.0, 5.0)
        self.cor = cor or random.choice([(0, 255, 140), (0, 220, 255), (255, 180, 0), (255, 255, 255)])
        self.raio = random.randint(3, 6)
        self.vida = 1.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.35  # gravidade
        self.vida -= 0.035
        return self.vida > 0

    def draw(self, img):
        if self.vida <= 0:
            return
        pt = (int(self.x), int(self.y))
        cv2.circle(img, pt, max(1, int(self.raio * self.vida)), self.cor, -1, cv2.LINE_AA)

# =============================================================================
# UTILITÁRIOS VISUAIS (GLASSMORPHISM CYBER-CLEAN)
# =============================================================================
def desenhar_retangulo_arredondado(img, pt1, pt2, cor_fundo, cor_borda, raio=14, alpha=0.85, espessura_borda=1):
    """Desenha card translúcido de alta performance operando direto na ROI."""
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
        c_len = min(16, w // 4, h // 4)
        c_cor = (min(255, cor_borda[0] + 50), min(255, cor_borda[1] + 50), min(255, cor_borda[2] + 50))
        cv2.line(img, (x1, y1), (x1 + c_len, y1), c_cor, 2, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + c_len), c_cor, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - c_len, y1), c_cor, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + c_len), c_cor, 2, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + c_len, y2), c_cor, 2, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - c_len), c_cor, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - c_len, y2), c_cor, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), c_cor, 2, cv2.LINE_AA)

    return img

def encontrar_camera():
    for idx in [1, 0, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[OK] Câmera encontrada no índice: {idx}")
                return cap
            cap.release()
    return None

# =============================================================================
# ANÁLISE ANATÔMICA AVANÇADA DOS DEDOS E GESTOS
# =============================================================================
def analisar_dedos_mao(hand_landmarks, is_right_hand, w, h):
    """
    Analisa os 21 pontos biométricos da mão.
    Retorna:
      - dedos_up: lista de 5 booleanos [Polegar, Indicador, Médio, Anelar, Mindinho]
      - pontas_coordenadas: lista de (x, y) das pontas dos dedos erguidos
      - gesto_nome: string com nome do gesto reconhecido
    """
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    escala = max(10.0, math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]))

    dedos_up = [False, False, False, False, False]
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    # 1. 4 Dedos Principais (Indicador, Médio, Anelar, Mindinho)
    for i, (tip, pip) in enumerate(zip(tip_ids, pip_ids)):
        # Considera erguido se a ponta estiver visivelmente acima da articulação PIP
        if pts[tip][1] < pts[pip][1] - (escala * 0.10):
            dedos_up[i + 1] = True

    # 2. Polegar: distância lateral e afastamento em relação ao ponto 2 e ponto 17
    dist_p4_p17 = math.hypot(pts[4][0] - pts[17][0], pts[4][1] - pts[17][1]) / escala
    dist_p4_p2 = math.hypot(pts[4][0] - pts[2][0], pts[4][1] - pts[2][1]) / escala
    if dist_p4_p17 > 0.65 and dist_p4_p2 > 0.35:
        dedos_up[0] = True

    pontas_coordenadas = []
    ids_pontas = [4, 8, 12, 16, 20]
    for i, erguido in enumerate(dedos_up):
        if erguido:
            pontas_coordenadas.append(pts[ids_pontas[i]])

    # 3. Reconhecimento de Gestos Especiais
    total_up = sum(dedos_up)
    gesto_nome = ""

    if total_up == 0:
        gesto_nome = "PUNHO FECHADO"
    elif total_up == 5:
        gesto_nome = "PALMA ABERTA"
    elif total_up == 1:
        if dedos_up[1]:
            gesto_nome = "APONTANDO (1)"
        elif dedos_up[0]:
            gesto_nome = "JOINHA (POSITIVO)"
        elif dedos_up[4]:
            gesto_nome = "MINDINHO (PROMESSA)"
    elif total_up == 2:
        if dedos_up[1] and dedos_up[2]:
            gesto_nome = "PAZ E AMOR (VITORIA)"
        elif dedos_up[0] and dedos_up[4]:
            gesto_nome = "HANG LOOSE (SHAKA)"
        elif dedos_up[1] and dedos_up[4]:
            gesto_nome = "ROCK 'N' ROLL (METAL)"
        elif dedos_up[0] and dedos_up[1]:
            gesto_nome = "LETRA L / PISTOLA"
    elif total_up == 3:
        if dedos_up[1] and dedos_up[2] and dedos_up[3]:
            gesto_nome = "TRES DEDOS"
        elif dedos_up[0] and dedos_up[1] and dedos_up[4]:
            gesto_nome = "I LOVE YOU (LIBRAS)"
    elif total_up == 4:
        if not dedos_up[0]:
            gesto_nome = "QUATRO DEDOS"

    return dedos_up, pontas_coordenadas, gesto_nome, pts

# =============================================================================
# DESENHO DO HUD HOLOGRÁFICO BIOMÉTRICO
# =============================================================================
def desenhar_mira_holografica(img, pt, tempo_anim):
    """Desenha retícula holográfica giratória na ponta do dedo erguido."""
    x, y = pt
    raio_base = 14
    cv2.circle(img, (x, y), raio_base, (0, 255, 140), 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 4, (0, 220, 255), -1, cv2.LINE_AA)

    # 4 arcos/linhas de mira em rotação
    ang_offset = tempo_anim * 3.0
    for i in range(4):
        ang = ang_offset + i * (math.pi / 2)
        x_m = int(x + math.cos(ang) * (raio_base + 6))
        y_m = int(y + math.sin(ang) * (raio_base + 6))
        cv2.circle(img, (x_m, y_m), 2, (0, 220, 255), -1, cv2.LINE_AA)

# =============================================================================
# MOTOR DO JOKENPÔ
# =============================================================================
OPCOES_JOKENPO = ["PEDRA", "PAPEL", "TESOURA"]

def converter_dedos_em_jokenpo(total_dedos, dedos_up):
    """Traduz a pose da mão em uma jogada de Pedra, Papel e Tesoura."""
    if total_dedos == 0:
        return "PEDRA"
    elif total_dedos == 5:
        return "PAPEL"
    elif total_dedos == 2 and (dedos_up[1] and dedos_up[2]):
        return "TESOURA"
    return None

def avaliar_vencedor_jokenpo(jogador, robo):
    if jogador == robo:
        return "EMPATE"
    regras = {
        ("PEDRA", "TESOURA"): "VITORIA",
        ("TESOURA", "PAPEL"): "VITORIA",
        ("PAPEL", "PEDRA"): "VITORIA"
    }
    if (jogador, robo) in regras:
        return "VITORIA"
    return "DERROTA"

# =============================================================================
# FLUXO PRINCIPAL DO FINGER ARCADE
# =============================================================================
def main():
    print("Iniciando Finger Arcade & Gesture Arena (ADS Unimar Aberta)...")
    cap = encontrar_camera()
    if cap is None:
        print("[ERRO] Nenhuma webcam compatível encontrada!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55
    )

    nome_janela = "Finger Arcade & Gesture Arena | ADS UNIMAR ABERTA"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    fullscreen = False
    espelhar_video = True

    # 3 Modos:
    # 0 = DASHBOARD BIOMÉTRICO (F1)
    # 1 = MATH BLITZ ARCADE (F2)
    # 2 = JOKENPÔ CONTRA A IA (F3)
    modo_atual = 0
    nomes_modos = ["DASHBOARD BIOMETRICO SCI-FI", "MATH BLITZ & REACAO RAPIDA", "JOKENPO CONTRA O ROBO DE ADS"]

    particulas = []
    fps_tempo = time.time()
    fps_cont = 0
    fps_display = 0

    # ==========================
    # VARIÁVEIS DO MODO 2 (MATH BLITZ)
    # ==========================
    math_score = 0
    math_combo = 1
    math_recorde = carregar_recorde()
    math_tempo_restante = 45.0
    math_jogo_ativo = False
    math_game_over = False
    math_tempo_inicio = 0
    math_pergunta_txt = "PREPARE-SE..."
    math_resposta_alvo = 0
    math_tempo_acerto_sustentado = 0
    math_anim_acerto_timer = 0
    math_feedback_txt = ""

    def sortear_desafio_math():
        nonlocal math_pergunta_txt, math_resposta_alvo
        tipo = random.choice(["direto", "soma", "sub"])
        if tipo == "direto":
            math_resposta_alvo = random.randint(1, 10)
            math_pergunta_txt = f"MOSTRE: {math_resposta_alvo} DEDOS!"
        elif tipo == "soma":
            a = random.randint(1, 5)
            b = random.randint(1, 5)
            math_resposta_alvo = a + b
            math_pergunta_txt = f"QUANTO E: {a} + {b} = ?"
        else:
            a = random.randint(4, 10)
            b = random.randint(1, a - 1)
            math_resposta_alvo = a - b
            math_pergunta_txt = f"QUANTO E: {a} - {b} = ?"

    # ==========================
    # VARIÁVEIS DO MODO 3 (JOKENPÔ)
    # ==========================
    jkp_estado = "AGUARDANDO"  # AGUARDANDO, CONTAGEM, RESULTADO
    jkp_tempo_fase = 0
    jkp_contagem_num = 3
    jkp_ultimo_som_tick = 0
    jkp_jogada_robo = ""
    jkp_jogada_jogador = ""
    jkp_resultado_txt = ""
    jkp_vitorias = 0
    jkp_derrotas = 0
    jkp_empates = 0
    jkp_streak = 0

    while True:
        success, img = cap.read()
        if not success or img is None:
            break

        if espelhar_video:
            img = cv2.flip(img, 1)

        h, w = img.shape[:2]
        now = time.time()

        # OTIMIZAÇÃO CRÍTICA DE FPS: Inferência MediaPipe em 640x360
        img_small = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        total_dedos_global = 0
        dedos_esq_count = 0
        dedos_dir_count = 0
        gestos_detectados = []
        pontas_holograficas = []
        pose_jokenpo_jogador = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label_raw = handedness_info.classification[0].label
                is_right_hand = (label_raw == "Right") if espelhar_video else (label_raw == "Left")

                dedos_up, pontas, gesto, pts_raw = analisar_dedos_mao(hand_landmarks, is_right_hand, w, h)
                qtd_dedos = sum(dedos_up)
                total_dedos_global += qtd_dedos
                pontas_holograficas.extend(pontas)

                if is_right_hand:
                    dedos_dir_count = qtd_dedos
                else:
                    dedos_esq_count = qtd_dedos

                if gesto:
                    rotulo_mao = "DIR" if is_right_hand else "ESQ"
                    gestos_detectados.append(f"{rotulo_mao}: {gesto}")

                # Jokenpô captura a primeira mão com gesto válido
                if pose_jokenpo_jogador is None:
                    jkp_cand = converter_dedos_em_jokenpo(qtd_dedos, dedos_up)
                    if jkp_cand:
                        pose_jokenpo_jogador = jkp_cand

                # Desenho do Esqueleto Biométrico Neon
                conexoes = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17)
                ]
                cor_linha = (0, 220, 255) if is_right_hand else (255, 180, 0)
                for p1, p2 in conexoes:
                    cv2.line(img, pts_raw[p1], pts_raw[p2], cor_linha, 1, cv2.LINE_AA)
                for p_idx, pt in enumerate(pts_raw):
                    cv2.circle(img, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Atualiza e desenha partículas
        particulas = [p for p in particulas if p.update()]
        for p in particulas:
            p.draw(img)

        # ---------------------------------------------------------------------
        # MODO 0: DASHBOARD BIOMÉTRICO SCI-FI
        # ---------------------------------------------------------------------
        if modo_atual == 0:
            # Retículas holográficas nas pontas dos dedos erguidos
            for pt in pontas_holograficas:
                desenhar_mira_holografica(img, pt, now)

            # Card Superior Central de Contagem Total
            card_w = 460
            card_h = 110
            cx = w // 2 - card_w // 2
            cy = 20

            desenhar_retangulo_arredondado(
                img, (cx, cy), (cx + card_w, cy + card_h),
                cor_fundo=(12, 14, 24), cor_borda=(0, 220, 255), raio=14, alpha=0.90, espessura_borda=2
            )
            cv2.putText(
                img, "DEDOS ERGUIDOS (AMBAS AS MAOS)", (cx + 55, cy + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 220, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, str(total_dedos_global), (cx + card_w // 2 - 25, cy + 92),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 140), 3, cv2.LINE_AA
            )

            # Cards Laterais com Contagem Individual
            # Mão Esquerda
            desenhar_retangulo_arredondado(
                img, (25, 25), (230, 95),
                cor_fundo=(12, 14, 24), cor_borda=(255, 180, 0), raio=12, alpha=0.88, espessura_borda=1
            )
            cv2.putText(img, "MAO ESQUERDA", (38, 48), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 180, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"{dedos_esq_count} dedos", (38, 80), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            # Mão Direita
            desenhar_retangulo_arredondado(
                img, (w - 230, 25), (w - 25, 95),
                cor_fundo=(12, 14, 24), cor_borda=(0, 220, 255), raio=12, alpha=0.88, espessura_borda=1
            )
            cv2.putText(img, "MAO DIREITA", (w - 215, 48), cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"{dedos_dir_count} dedos", (w - 215, 80), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            # Badge do Gesto Reconhecido
            if gestos_detectados:
                texto_gesto = " | ".join(gestos_detectados)
                gw = min(500, len(texto_gesto) * 14 + 40)
                desenhar_retangulo_arredondado(
                    img, (w // 2 - gw // 2, cy + card_h + 12), (w // 2 + gw // 2, cy + card_h + 52),
                    cor_fundo=(15, 25, 40), cor_borda=(0, 255, 140), raio=10, alpha=0.92, espessura_borda=1
                )
                cv2.putText(
                    img, texto_gesto, (w // 2 - gw // 2 + 18, cy + card_h + 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 140), 1, cv2.LINE_AA
                )

        # ---------------------------------------------------------------------
        # MODO 1: MATH BLITZ & REAÇÃO RÁPIDA
        # ---------------------------------------------------------------------
        elif modo_atual == 1:
            if not math_jogo_ativo and not math_game_over:
                # Tela de Entrada do Jogo
                cx_start = w // 2 - 320
                cy_start = h // 2 - 140
                desenhar_retangulo_arredondado(
                    img, (cx_start, cy_start), (cx_start + 640, cy_start + 260),
                    cor_fundo=(10, 14, 24), cor_borda=(0, 255, 140), raio=16, alpha=0.94, espessura_borda=2
                )
                cv2.putText(img, "MATH BLITZ * DESAFIO DE DEDOS", (cx_start + 45, cy_start + 48),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 140), 2, cv2.LINE_AA)
                cv2.line(img, (cx_start + 40, cy_start + 68), (cx_start + 600, cy_start + 68), (45, 55, 75), 1)
                cv2.putText(img, "Responda as perguntas mostrando a quantidade de dedos!", (cx_start + 45, cy_start + 110),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, (220, 225, 235), 1, cv2.LINE_AA)
                cv2.putText(img, "Tempo total de jogo: 45 segundos de pura agilidade!", (cx_start + 45, cy_start + 145),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"Recorde Atual do Stand: {math_recorde} PONTOS", (cx_start + 45, cy_start + 185),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(img, "Pressione [ESPACO] ou [R] para Iniciar a Partida!", (cx_start + 45, cy_start + 230),
                            cv2.FONT_HERSHEY_DUPLEX, 0.54, (0, 255, 140), 1, cv2.LINE_AA)

            elif math_jogo_ativo:
                tempo_decorrido = now - math_tempo_inicio
                tempo_restante = max(0.0, math_tempo_restante - tempo_decorrido)

                if tempo_restante <= 0:
                    math_jogo_ativo = False
                    math_game_over = True
                    if math_score > math_recorde:
                        math_recorde = math_score
                        salvar_recorde(math_recorde)
                        tocar_som_vitoria()
                    else:
                        tocar_som_derrota()

                # Card Superior da Pergunta
                card_w = 600
                card_h = 135
                cx = w // 2 - card_w // 2
                cy = 20

                cor_borda_p = (0, 255, 140) if (now < math_anim_acerto_timer) else (0, 220, 255)
                desenhar_retangulo_arredondado(
                    img, (cx, cy), (cx + card_w, cy + card_h),
                    cor_fundo=(12, 14, 24), cor_borda=cor_borda_p, raio=14, alpha=0.92, espessura_borda=2
                )
                cv2.putText(
                    img, f"SCORE: {math_score}   |   COMBO: x{math_combo}   |   TEMPO: {int(tempo_restante)}s",
                    (cx + 35, cy + 32), cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 200, 0), 1, cv2.LINE_AA
                )
                cv2.putText(
                    img, math_pergunta_txt, (cx + 35, cy + 85),
                    cv2.FONT_HERSHEY_DUPLEX, 0.92, (255, 255, 255), 2, cv2.LINE_AA
                )

                # Feedback do jogador
                cv2.putText(
                    img, f"Voce esta mostrando: {total_dedos_global} dedos", (cx + 35, cy + 118),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA
                )

                # Validação da Resposta
                if total_dedos_global == math_resposta_alvo:
                    if math_tempo_acerto_sustentado == 0:
                        math_tempo_acerto_sustentado = now
                    elif now - math_tempo_acerto_sustentado >= 0.35:
                        # Acertou!
                        pontos_ganhos = 100 * math_combo
                        math_score += pontos_ganhos
                        math_combo = min(5, math_combo + 1)
                        math_anim_acerto_timer = now + 0.6
                        tocar_som_acerto()

                        # Explode partículas nas pontas dos dedos
                        for pt in pontas_holograficas:
                            for _ in range(6):
                                particulas.append(Particula(pt[0], pt[1]))

                        sortear_desafio_math()
                        math_tempo_acerto_sustentado = 0
                else:
                    math_tempo_acerto_sustentado = 0

            elif math_game_over:
                # Tela de Game Over
                cx_go = w // 2 - 320
                cy_go = h // 2 - 140
                desenhar_retangulo_arredondado(
                    img, (cx_go, cy_go), (cx_go + 640, cy_go + 260),
                    cor_fundo=(12, 14, 24), cor_borda=(0, 255, 140), raio=16, alpha=0.94, espessura_borda=2
                )
                cv2.putText(img, "FIM DE JOGO!", (cx_go + 200, cy_go + 52),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 140), 2, cv2.LINE_AA)
                cv2.line(img, (cx_go + 40, cy_go + 72), (cx_go + 600, cy_go + 72), (45, 55, 75), 1)

                cv2.putText(img, f"SUA PONTUACAO FINAL: {math_score} PONTOS", (cx_go + 45, cy_go + 125),
                            cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f"MELHOR RECORDE DO STAND: {math_recorde} PONTOS", (cx_go + 45, cy_go + 170),
                            cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(img, "Pressione [ESPACO] ou [R] para Jogar Novamente!", (cx_go + 45, cy_go + 225),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # MODO 2: JOKENPÔ CONTRA O ROBÔ DE ADS
        # ---------------------------------------------------------------------
        elif modo_atual == 2:
            card_w = 640
            card_h = 170
            cx = w // 2 - card_w // 2
            cy = 20

            desenhar_retangulo_arredondado(
                img, (cx, cy), (cx + card_w, cy + card_h),
                cor_fundo=(12, 14, 24), cor_borda=(0, 220, 255), raio=14, alpha=0.92, espessura_borda=2
            )
            cv2.putText(
                img, f"VITORIAS: {jkp_vitorias}  |  EMPATES: {jkp_empates}  |  ROBO ADS: {jkp_derrotas}  (Streak: {jkp_streak})",
                (cx + 35, cy + 32), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 200, 0), 1, cv2.LINE_AA
            )

            if jkp_estado == "AGUARDANDO":
                cv2.putText(
                    img, "PREPARE SUA MAO: PEDRA, PAPEL OU TESOURA", (cx + 35, cy + 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.putText(
                    img, "Pressione [ESPACO] para iniciar o duelo!", (cx + 35, cy + 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 140), 2, cv2.LINE_AA
                )

            elif jkp_estado == "CONTAGEM":
                tempo_decorrido = now - jkp_tempo_fase
                if tempo_decorrido < 1.0:
                    cnt_txt = "3..."
                elif tempo_decorrido < 2.0:
                    cnt_txt = "2..."
                elif tempo_decorrido < 3.0:
                    cnt_txt = "1..."
                else:
                    cnt_txt = "JA!"

                # Toca som de tick a cada segundo
                seg_atual = int(tempo_decorrido)
                if seg_atual != jkp_ultimo_som_tick and seg_atual < 3:
                    tocar_som_tick()
                    jkp_ultimo_som_tick = seg_atual

                cv2.putText(
                    img, f"CONTAGEM: {cnt_txt}", (cx + 140, cy + 98),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 140), 3, cv2.LINE_AA
                )

                if tempo_decorrido >= 3.3:
                    # Avalia o resultado
                    jkp_jogada_robo = random.choice(OPCOES_JOKENPO)
                    jkp_jogada_jogador = pose_jokenpo_jogador or "INDEFINIDO"

                    if jkp_jogada_jogador == "INDEFINIDO":
                        jkp_resultado_txt = "NAO DETECTOU GESTO VALIDO"
                        tocar_som_erro()
                    else:
                        res = avaliar_vencedor_jokenpo(jkp_jogada_jogador, jkp_jogada_robo)
                        if res == "VITORIA":
                            jkp_vitorias += 1
                            jkp_streak += 1
                            jkp_resultado_txt = "VOCE VENCEU O ROBO DE ADS!"
                            tocar_som_vitoria()
                            # Partículas de comemoração
                            for _ in range(40):
                                particulas.append(Particula(w // 2, h // 2))
                        elif res == "DERROTA":
                            jkp_derrotas += 1
                            jkp_streak = 0
                            jkp_resultado_txt = "O ROBO DE ADS VENCEU!"
                            tocar_som_derrota()
                        else:
                            jkp_empates += 1
                            jkp_resultado_txt = "EMPATE!"
                            tocar_som_empate()

                    jkp_estado = "RESULTADO"
                    jkp_tempo_fase = now

            elif jkp_estado == "RESULTADO":
                cor_res = (0, 255, 140) if "VENCEU" in jkp_resultado_txt and "VOCE" in jkp_resultado_txt else (
                    (0, 100, 255) if "ROBO" in jkp_resultado_txt else (255, 200, 0)
                )
                cv2.putText(
                    img, jkp_resultado_txt, (cx + 35, cy + 72),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, cor_res, 2, cv2.LINE_AA
                )
                cv2.putText(
                    img, f"Voce: {jkp_jogada_jogador}   vs   Robo: {jkp_jogada_robo}", (cx + 35, cy + 115),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.putText(
                    img, "[ESPACO] para a Proxima Rodada!", (cx + 35, cy + 152),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 220, 255), 1, cv2.LINE_AA
                )

                # Próxima rodada automática após 4 segundos
                if now - jkp_tempo_fase > 4.0:
                    jkp_estado = "AGUARDANDO"

        # ---------------------------------------------------------------------
        # RODAPÉ COM ATALHOS E INDICAÇÃO DO MODO (38px Cyber-Clean)
        # ---------------------------------------------------------------------
        foot_h = 38
        foot_roi = img[h - foot_h:h, 0:w]
        foot_bg = np.full(foot_roi.shape, (10, 12, 18), dtype=np.uint8)
        cv2.addWeighted(foot_bg, 0.88, foot_roi, 0.12, 0, foot_roi)

        cv2.putText(
            img, f"[F1/F2/F3] MODO: {nomes_modos[modo_atual]}", (20, h - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 140), 1, cv2.LINE_AA
        )

        status_espelho = "LIGADO" if espelhar_video else "DESLIGADO"
        cor_espelho = (0, 255, 140) if espelhar_video else (0, 220, 255)
        cv2.putText(
            img, f"[ESPACO] Espelho: {status_espelho}", (375, h - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.44, cor_espelho, 1, cv2.LINE_AA
        )

        cv2.putText(
            img, "[R: Reiniciar] | [TAB/F: Tela Cheia] | [ESC: Sair]",
            (w // 2 + 80, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.44, (180, 185, 200), 1, cv2.LINE_AA
        )

        fps_cont += 1
        if time.time() - fps_tempo >= 1.0:
            fps_display = fps_cont
            fps_cont = 0
            fps_tempo = time.time()

        cv2.putText(
            img, f"{fps_display} FPS", (w - 85, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 140), 1, cv2.LINE_AA
        )

        cv2.imshow(nome_janela, img)
        key_raw = cv2.waitKeyEx(1)
        if key_raw == -1:
            key = -1
        else:
            key = key_raw & 0xFF

        # Detecção de teclas especiais F1, F2, F3
        is_f1 = (key_raw in [7340032, 0x700000, 65470]) or ((key_raw >> 16) == 0x70)
        is_f2 = (key_raw in [7405568, 0x710000, 65471]) or ((key_raw >> 16) == 0x71)
        is_f3 = (key_raw in [7471104, 0x720000, 65472]) or ((key_raw >> 16) == 0x72)

        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC / Q
            break
        elif is_f1:
            modo_atual = 0
        elif is_f2:
            modo_atual = 1
            if not math_jogo_ativo and not math_game_over:
                sortear_desafio_math()
        elif is_f3:
            modo_atual = 2
            jkp_estado = "AGUARDANDO"
        elif key == 32:  # ESPAÇO: Ação contextual ou Espelho
            if modo_atual == 1:
                if not math_jogo_ativo or math_game_over:
                    math_jogo_ativo = True
                    math_game_over = False
                    math_score = 0
                    math_combo = 1
                    math_tempo_inicio = time.time()
                    sortear_desafio_math()
                else:
                    espelhar_video = not espelhar_video
            elif modo_atual == 2:
                if jkp_estado in ["AGUARDANDO", "RESULTADO"]:
                    jkp_estado = "CONTAGEM"
                    jkp_tempo_fase = time.time()
                    jkp_ultimo_som_tick = -1
                else:
                    espelhar_video = not espelhar_video
            else:
                espelhar_video = not espelhar_video
        elif key == ord('r') or key == ord('R'):  # R: Reiniciar Jogo
            if modo_atual == 1:
                math_jogo_ativo = True
                math_game_over = False
                math_score = 0
                math_combo = 1
                math_tempo_inicio = time.time()
                sortear_desafio_math()
            elif modo_atual == 2:
                jkp_estado = "CONTAGEM"
                jkp_tempo_fase = time.time()
                jkp_ultimo_som_tick = -1
        elif key == 9 or key == ord('f') or key == ord('F'):  # TAB / F: Tela Cheia
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
