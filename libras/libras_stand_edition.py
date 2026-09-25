"""
=============================================================================
     LIBRAS EDUCATIONAL AI - STAND EDITION (ADS UNIMAR ABERTA)
=============================================================================
Modos de Tela (Alternáveis com a Tecla [M]):
- Modo 0: TELA LIMPA (Minimalista / Zen):
  Apenas a imagem da câmera e uma pílula flutuante moderna com a letra reconhecida.
- Modo 1: MODO DESAFIO (Jogo Educacional do Stand):
  Card completo com meta de letra, instruções anatômicas oficiais de LIBRAS,
  validador de 1s e placar de acertos.
- Modo 2: MODO SOLETRADOR (Escreva seu Nome no Ar):
  Segure uma letra por 1s e ela é adicionada à palavra na tela (ex: L-U-C-A-S)!

Easter Eggs:
- Dedo do Meio: Efeito censura de TV (pixelate + tarja vermelha + som PIIII).
- Joinha: Chuva de confetes dourados/neon + selo de aprovação do stand!

Atalhos do Teclado:
- [F1]            = Modo 0: TELA LIMPA (Minimalista / Zen)
- [F2]            = Modo 1: MODO DESAFIO (Jogo Educacional do Stand)
- [F3]            = Modo 2: MODO SOLETRADOR (Escreva seu Nome no Ar)
- [ESPAÇO]        = Inverter Espelho da Câmera (Iriun)
- [TAB]           = Alternar Tela Cheia
- [A a Z]         = Ensinar / Calibrar qualquer letra em 2 segundos (todas as 26 letras livres!)
- [1]             = Calibrar Joinha
- [2]             = Calibrar Censura
- [BACKSPACE]     = Apagar última letra (no Modo Soletrar)
- [DELETE]        = Limpar palavra inteira (no Modo Soletrar)
- [ESC]           = Sair
=============================================================================
"""

import cv2
import numpy as np
import os
import sys
import time
import random
import threading
import mediapipe as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from libras_ml_engine import LibrasMLEngine, DICAS_EDUCACIONAIS

# Efeitos Sonoros Arcade & TV
try:
    import winsound
    def tocar_som_acerto():
        winsound.Beep(988, 70)
        winsound.Beep(1318, 110)
    def tocar_som_gravar():
        winsound.Beep(650, 45)
        winsound.Beep(950, 75)
    def tocar_som_letra_adicionada():
        winsound.Beep(784, 60)
        winsound.Beep(1046, 80)
    def tocar_som_joinha():
        threading.Thread(target=_worker_joinha, daemon=True).start()
    def _worker_joinha():
        for freq in [523, 659, 784, 1046]:
            winsound.Beep(freq, 50)
    def tocar_som_piii():
        threading.Thread(target=_worker_piii, daemon=True).start()
    def _worker_piii():
        winsound.Beep(1000, 220)
except Exception:
    def tocar_som_acerto(): pass
    def tocar_som_gravar(): pass
    def tocar_som_letra_adicionada(): pass
    def tocar_som_joinha(): pass
    def tocar_som_piii(): pass

# =============================================================================
# SISTEMA DE CONFETES / PARTÍCULAS PARA O JOINHA
# =============================================================================
class Confete:
    def __init__(self, w, h):
        self.x = random.randint(20, w - 20)
        self.y = random.randint(-60, -10)
        self.vy = random.uniform(6.0, 14.0)
        self.vx = random.uniform(-3.0, 3.0)
        self.cor = random.choice([
            (0, 255, 140), (0, 220, 255), (0, 215, 255), (255, 120, 220), (255, 255, 255)
        ])
        self.raio = random.randint(3, 7)

    def update(self, h):
        self.y += self.vy
        self.x += self.vx
        return self.y < h

    def draw(self, img):
        pt = (int(self.x), int(self.y))
        cv2.circle(img, pt, self.raio, self.cor, -1, cv2.LINE_AA)

# =============================================================================
# UTILITÁRIOS VISUAIS (GLASSMORPHISM & CENSURA)
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

def aplicar_censura_pixelate(img, bbox, tamanho_bloco=14):
    x1, y1, x2, y2 = bbox
    pad = 25
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return

    sub = img[y1:y2, x1:x2]
    h_sub, w_sub = sub.shape[:2]
    small = cv2.resize(sub, (max(1, w_sub // tamanho_bloco), max(1, h_sub // tamanho_bloco)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w_sub, h_sub), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pixelated

    tarja_h = 42
    my = (y1 + y2) // 2
    cv2.rectangle(img, (x1 - 10, my - tarja_h // 2), (x2 + 10, my + tarja_h // 2), (10, 10, 15), -1)
    cv2.rectangle(img, (x1 - 10, my - tarja_h // 2), (x2 + 10, my + tarja_h // 2), (0, 0, 255), 2)
    cv2.putText(
        img, "[ CENSURADO ]", (x1 + 10, my + 8),
        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA
    )

def desenhar_landmarks_biometricos(img, hand_landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    conexoes = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]
    for p1_idx, p2_idx in conexoes:
        cv2.line(img, pts[p1_idx], pts[p2_idx], (220, 225, 235), 1, cv2.LINE_AA)

    pontas = [4, 8, 12, 16, 20]
    for i, pt in enumerate(pts):
        if i in pontas:
            cv2.circle(img, pt, 7, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.circle(img, pt, 4, (0, 255, 140), -1, cv2.LINE_AA)
            cv2.circle(img, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, pt, 3, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt, 1, (255, 255, 255), -1, cv2.LINE_AA)

def desenhar_landmarks_mao_esquerda_aviso(img, hand_landmarks, w, h):
    """Desenha landmarks da mão esquerda em tom de aviso âmbar para orientar o usuário a usar a direita."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    conexoes = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]
    for p1_idx, p2_idx in conexoes:
        cv2.line(img, pts[p1_idx], pts[p2_idx], (0, 140, 255), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(img, pt, 3, (0, 180, 255), -1, cv2.LINE_AA)

def encontrar_camera():
    for idx in [1, 0, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[OK] Câmera conectada no índice: {idx}")
                return cap
            cap.release()
    return None

def main():
    print("Iniciando IA Educacional de LIBRAS (ADS Unimar)...")
    cap = encontrar_camera()
    if cap is None:
        print("[ERRO] Nenhuma webcam encontrada!")
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

    ml_engine = LibrasMLEngine()
    nome_janela = "IA Educacional LIBRAS | ADS UNIMAR ABERTA"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    fullscreen = False
    espelhar_video = False

    # Modos de Exibição:
    # 0 = TELA LIMPA (Apenas a letra no cantinho, sem poluição)
    # 1 = MODO JOGO DESAFIO (Com metas, pontuação e dicas)
    # 2 = MODO SOLETRADOR (Construa palavras letra por letra no ar)
    modo_tela = 1
    nomes_modos = ["TELA LIMPA (MINIMALISTA)", "JOGO DESAFIO EDUCATIVO", "SOLETRADOR DE PALAVRAS"]

    # Variáveis do Modo Desafio
    letras_desafio = ["A", "B", "C", "D", "E", "I", "L", "O", "R", "S", "U", "V", "W", "Y"]
    meta_letra = random.choice(letras_desafio)
    score_desafio = 0
    tempo_acerto_inicio = None
    tempo_necessario = 1.0
    anim_acerto_timer = 0

    # Variáveis do Modo Soletrador
    palavra_soletrada = ""
    tempo_soletrar_inicio = None
    ultima_letra_soletrada = ""

    # Modo de Calibração / Gravação
    modo_gravando = False
    letra_alvo_gravacao = ""
    frames_gravados = 0
    total_frames_gravar = 20
    calibrado_feedback_txt = ""
    calibrado_feedback_timer = 0.0

    # Easter Eggs
    easter_egg_censura_ativo = False
    easter_egg_joinha_ativo = False
    confetes = []
    ultimo_som_piii = 0
    ultimo_som_joinha = 0

    fps_tempo = time.time()
    fps_cont = 0
    fps_display = 0

    while True:
        success, img = cap.read()
        if not success or img is None:
            break

        if espelhar_video:
            img = cv2.flip(img, 1)

        h, w = img.shape[:2]

        # OTIMIZAÇÃO CRÍTICA DE FPS:
        # Passa imagem redimensionada (640x360) para inferência do MediaPipe.
        # Os landmarks de saída são normalizados (0.0 a 1.0), gerando a mesma
        # precisão geométrica na tela de 1280x720 com custo computacional 4x menor!
        img_small = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        letra_detectada = "---"
        confianca_val = 0.0
        dica_letra = "Posicione a mao em frente a camera"
        hand_found = False
        lmList = []
        hand_bbox = None

        easter_egg_censura_ativo = False
        easter_egg_joinha_ativo = False

        mao_direita_encontrada = False
        mao_esquerda_apenas = False
        hand_landmarks_escolhida = None

        if results.multi_hand_landmarks and results.multi_handedness:
            # 1. Busca prioritária exclusiva pela Mão DIREITA física
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label_raw = handedness_info.classification[0].label
                # Com espelhamento horizontal ativado (selfie flip): 'Right' no MediaPipe é a Mão Direita física
                # Sem espelhamento: 'Left' no MediaPipe é a Mão Direita física
                is_right_hand = (label_raw == "Right") if espelhar_video else (label_raw == "Left")

                if is_right_hand:
                    mao_direita_encontrada = True
                    hand_landmarks_escolhida = hand_landmarks
                    break

            # Se nenhuma mão direita foi encontrada, mas há mão esquerda em frente à câmera
            if not mao_direita_encontrada and len(results.multi_hand_landmarks) > 0:
                mao_esquerda_apenas = True
                desenhar_landmarks_mao_esquerda_aviso(img, results.multi_hand_landmarks[0], w, h)
                dica_letra = "ATENCAO: Mostre a mao DIREITA para LIBRAS"

            # 2. Se a mão direita foi confirmada, processa calibração e classificação
            if mao_direita_encontrada and hand_landmarks_escolhida is not None:
                hand_found = True
                pts_x = [int(lm.x * w) for lm in hand_landmarks_escolhida.landmark]
                pts_y = [int(lm.y * h) for lm in hand_landmarks_escolhida.landmark]
                hand_bbox = (min(pts_x), min(pts_y), max(pts_x), max(pts_y))

                for id_pt, lm in enumerate(hand_landmarks_escolhida.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), lm.z
                    lmList.append([id_pt, px, py, pz])

                # Gravação de calibração (apenas com mão direita!)
                if modo_gravando and letra_alvo_gravacao:
                    sucesso_add = ml_engine.adicionar_amostra(letra_alvo_gravacao, lmList)
                    if sucesso_add:
                        frames_gravados += 1
                        if frames_gravados >= total_frames_gravar:
                            salvo_ok = ml_engine.finalizar_gravacao_classe(letra_alvo_gravacao)
                            tocar_som_gravar()
                            calibrado_feedback_txt = f"LETRA '{letra_alvo_gravacao}' CALIBRADA E SALVA NO DISCO!"
                            calibrado_feedback_timer = time.time() + 3.0
                            modo_gravando = False
                            letra_alvo_gravacao = ""
                            frames_gravados = 0

                # Classificação ML precisa
                letra_detectada, confianca_val, dica_letra = ml_engine.classificar(lmList)

                # Detecção Anatômica de Easter Eggs
                pts_arr = [(lm.x * w, lm.y * h) for lm in hand_landmarks_escolhida.landmark]

                # Escala biométrica de referência da mão (pulso ao nó médio)
                escala_mao = np.hypot(pts_arr[9][0] - pts_arr[0][0], pts_arr[9][1] - pts_arr[0][1])
                if escala_mao < 10.0:
                    escala_mao = 100.0

                # 1. Censura (Dedo do Meio 🖕)
                medio_ereto = (pts_arr[12][1] < pts_arr[10][1]) and (pts_arr[12][1] < pts_arr[9][1] - (escala_mao * 0.40))
                indicador_fechado = pts_arr[8][1] > pts_arr[6][1]
                anelar_fechado = pts_arr[16][1] > pts_arr[14][1]
                mindinho_fechado = pts_arr[20][1] > pts_arr[18][1]

                if (medio_ereto and indicador_fechado and anelar_fechado and mindinho_fechado) or (letra_detectada == "CENSURA"):
                    easter_egg_censura_ativo = True

                # 2. Joinha (👍) vs Letra 'A'
                dist_4_5 = np.hypot(pts_arr[4][0] - pts_arr[5][0], pts_arr[4][1] - pts_arr[5][1])
                polegar_afastado = (dist_4_5 / escala_mao) >= 0.55
                polegar_muito_alto = (pts_arr[4][1] < pts_arr[5][1] - (escala_mao * 0.30)) and (pts_arr[4][1] < pts_arr[9][1] - (escala_mao * 0.25))

                joinha_anatomico = (
                    polegar_afastado and
                    polegar_muito_alto and
                    indicador_fechado and
                    anelar_fechado and
                    mindinho_fechado and
                    not medio_ereto and
                    letra_detectada not in ["A", "S", "E", "C", "O"]
                )

                if joinha_anatomico or (letra_detectada == "JOINHA" and confianca_val >= 70.0):
                    easter_egg_joinha_ativo = True

                if not easter_egg_censura_ativo:
                    desenhar_landmarks_biometricos(img, hand_landmarks_escolhida, w, h)

        now = time.time()

        # Disparo dos Easter Eggs
        if easter_egg_censura_ativo and hand_bbox is not None:
            aplicar_censura_pixelate(img, hand_bbox, tamanho_bloco=14)
            if now - ultimo_som_piii > 0.22:
                tocar_som_piii()
                ultimo_som_piii = now

        if easter_egg_joinha_ativo:
            if len(confetes) < 80:
                for _ in range(8):
                    confetes.append(Confete(w, h))

            if now - ultimo_som_joinha > 2.5:
                tocar_som_joinha()
                ultimo_som_joinha = now

            cx_selo = w // 2 - 280
            cy_selo = 80
            desenhar_retangulo_arredondado(
                img, (cx_selo, cy_selo), (cx_selo + 560, cy_selo + 85),
                cor_fundo=(10, 30, 20), cor_borda=(0, 255, 140), raio=14, alpha=0.92, espessura_borda=2
            )
            cv2.putText(
                img, "100% APROVADO PELO STAND DE ADS!", (cx_selo + 30, cy_selo + 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 140), 2, cv2.LINE_AA
            )
            cv2.putText(
                img, "MANDOU BEM DEMAIS! VALEU PELA VISITA!", (cx_selo + 30, cy_selo + 68),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

        confetes = [c for c in confetes if c.update(h)]
        for c in confetes:
            c.draw(img)

        # Label e Cores de exibição
        if easter_egg_censura_ativo:
            lbl_show = "[X]"
            cor_glow = (0, 0, 255)
        elif easter_egg_joinha_ativo:
            lbl_show = "TOP!"
            cor_glow = (0, 255, 140)
        else:
            lbl_show = letra_detectada
            cor_glow = (0, 255, 140) if letra_detectada != "---" else (90, 95, 110)

        # =====================================================================
        # RENDERIZAÇÃO CONFORME O MODO ESCOLHIDO (Tecla M)
        # =====================================================================

        # ---------------------------------------------------------------------
        # MODO 0: TELA LIMPA (MINIMALISTA / ZEN)
        # ---------------------------------------------------------------------
        if modo_tela == 0:
            # Pílula flutuante moderna no canto superior direito
            badge_w = 230
            badge_h = 75
            bx = w - badge_w - 25
            by = 25
            desenhar_retangulo_arredondado(
                img, (bx, by), (bx + badge_w, by + badge_h),
                cor_fundo=(12, 14, 22), cor_borda=(0, 220, 255), raio=14, alpha=0.88, espessura_borda=1
            )

            if len(lbl_show) <= 2:
                cv2.putText(img, lbl_show if lbl_show != "---" else "---", (bx + 18, by + 56), cv2.FONT_HERSHEY_DUPLEX, 1.8, cor_glow if lbl_show != "---" else (90, 95, 110), 2, cv2.LINE_AA)
                pos_info_x = bx + 80
            else:
                cv2.putText(img, lbl_show, (bx + 16, by + 50), cv2.FONT_HERSHEY_DUPLEX, 1.05, cor_glow, 2, cv2.LINE_AA)
                pos_info_x = bx + 110

            if easter_egg_joinha_ativo or easter_egg_censura_ativo:
                cv2.putText(img, "EASTER EGG", (pos_info_x, by + 30), cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 255, 140) if easter_egg_joinha_ativo else (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(img, "ATIVADO!", (pos_info_x, by + 54), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "SINAL LIBRAS", (pos_info_x, by + 30), cv2.FONT_HERSHEY_DUPLEX, 0.38, (160, 165, 180), 1, cv2.LINE_AA)
                cv2.putText(img, f"{confianca_val:.0f}% precisao", (pos_info_x, by + 54), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # MODO 1: MODO JOGO DESAFIO (GAME DO STAND)
        # ---------------------------------------------------------------------
        elif modo_tela == 1:
            progresso_segurar = 0.0

            if hand_found and not modo_gravando and not easter_egg_censura_ativo and letra_detectada == meta_letra and confianca_val >= 70.0:
                if tempo_acerto_inicio is None:
                    tempo_acerto_inicio = now
                else:
                    decorrido = now - tempo_acerto_inicio
                    progresso_segurar = min(1.0, decorrido / tempo_necessario)
                    if decorrido >= tempo_necessario:
                        score_desafio += 1
                        tocar_som_acerto()
                        anim_acerto_timer = now + 1.2
                        outras = [l for l in letras_desafio if l != meta_letra]
                        meta_letra = random.choice(outras) if outras else meta_letra
                        tempo_acerto_inicio = None
            else:
                tempo_acerto_inicio = None

            hud_w = 410
            hud_h = 390
            hud_x = w - hud_w - 24
            hud_y = 24

            desenhar_retangulo_arredondado(
                img, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h),
                cor_fundo=(12, 14, 22), cor_borda=(0, 220, 255), raio=16, alpha=0.88, espessura_borda=1
            )

            cv2.putText(img, "ADS * UNIMAR ABERTA", (hud_x + 24, hud_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "JOGO DESAFIO LIBRAS", (hud_x + 24, hud_y + 58), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(img, (hud_x + 20, hud_y + 70), (hud_x + hud_w - 20, hud_y + 70), (45, 50, 68), 1, cv2.LINE_AA)

            cv2.putText(img, "SINAL IDENTIFICADO:", (hud_x + 24, hud_y + 94), cv2.FONT_HERSHEY_DUPLEX, 0.42, (160, 165, 180), 1, cv2.LINE_AA)

            if lbl_show != "---":
                cv2.putText(img, lbl_show, (hud_x + 30, hud_y + 165), cv2.FONT_HERSHEY_DUPLEX, 2.3, (0, 100, 50) if not easter_egg_censura_ativo else (0, 0, 160), 8, cv2.LINE_AA)
                cv2.putText(img, lbl_show, (hud_x + 30, hud_y + 165), cv2.FONT_HERSHEY_DUPLEX, 2.3, cor_glow, 3, cv2.LINE_AA)
            else:
                cv2.putText(img, "---", (hud_x + 30, hud_y + 165), cv2.FONT_HERSHEY_DUPLEX, 2.3, (90, 95, 110), 2, cv2.LINE_AA)

            cv2.putText(img, f"PRECISAO: {confianca_val:.0f}%", (hud_x + 165, hud_y + 116), cv2.FONT_HERSHEY_DUPLEX, 0.46, (210, 215, 225), 1, cv2.LINE_AA)
            bar_x = hud_x + 165
            bar_y = hud_y + 128
            bar_w = 215
            bar_h = 14
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (25, 28, 38), -1)
            fill_w = int(bar_w * (confianca_val / 100.0))
            if fill_w > 0:
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 220, 255), -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 65, 85), 1)

            cv2.line(img, (hud_x + 20, hud_y + 185), (hud_x + hud_w - 20, hud_y + 185), (45, 50, 68), 1, cv2.LINE_AA)

            cv2.putText(img, "DESAFIO DO STAND • APRENDA LIBRAS", (hud_x + 24, hud_y + 210), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "FACA O SINAL DA LETRA:", (hud_x + 24, hud_y + 242), cv2.FONT_HERSHEY_DUPLEX, 0.44, (220, 225, 235), 1, cv2.LINE_AA)
            cv2.putText(img, f"'{meta_letra}'", (hud_x + 215, hud_y + 252), cv2.FONT_HERSHEY_DUPLEX, 1.35, (0, 230, 255), 2, cv2.LINE_AA)

            dica_meta = DICAS_EDUCACIONAIS.get(meta_letra, "")
            p1 = dica_meta[:42]
            p2 = dica_meta[42:86]
            cv2.putText(img, p1, (hud_x + 24, hud_y + 282), cv2.FONT_HERSHEY_DUPLEX, 0.38, (175, 185, 205), 1, cv2.LINE_AA)
            if p2:
                cv2.putText(img, p2, (hud_x + 24, hud_y + 298), cv2.FONT_HERSHEY_DUPLEX, 0.38, (175, 185, 205), 1, cv2.LINE_AA)

            bar_val_w = int((hud_w - 48) * progresso_segurar)
            cv2.rectangle(img, (hud_x + 24, hud_y + 318), (hud_x + hud_w - 24, hud_y + 328), (25, 28, 38), -1)
            if bar_val_w > 0:
                cv2.rectangle(img, (hud_x + 24, hud_y + 318), (hud_x + 24 + bar_val_w, hud_y + 328), (0, 255, 180), -1)
            cv2.rectangle(img, (hud_x + 24, hud_y + 318), (hud_x + hud_w - 24, hud_y + 328), (60, 65, 85), 1)

            cv2.putText(
                img, f"ACERTOS DO ALUNO: {score_desafio}", (hud_x + 24, hud_y + 365),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 140), 2, cv2.LINE_AA
            )

            if now < anim_acerto_timer:
                cx_toast = w // 2 - 220
                cy_toast = h // 2 - 40
                desenhar_retangulo_arredondado(
                    img, (cx_toast, cy_toast), (cx_toast + 440, cy_toast + 80),
                    cor_fundo=(8, 20, 15), cor_borda=(0, 255, 140), raio=12, alpha=0.92, espessura_borda=2
                )
                cv2.putText(img, "PARABENS! VOCE APRENDEU O SINAL!", (cx_toast + 20, cy_toast + 48), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 140), 2, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # MODO 2: MODO SOLETRADOR DE NOMES (FORMAR PALAVRAS)
        # ---------------------------------------------------------------------
        elif modo_tela == 2:
            # Lógica para adicionar letra à palavra ao segurar estável por 1.0s
            prog_soletrar = 0.0
            if hand_found and letra_detectada not in ["---", "...", "CENSURA", "JOINHA"] and confianca_val >= 75.0:
                if letra_detectada != ultima_letra_soletrada:
                    ultima_letra_soletrada = letra_detectada
                    tempo_soletrar_inicio = now
                else:
                    decorrido = now - (tempo_soletrar_inicio or now)
                    prog_soletrar = min(1.0, decorrido / 1.0)
                    if decorrido >= 1.0:
                        if len(palavra_soletrada) < 18:
                            palavra_soletrada += letra_detectada
                            tocar_som_letra_adicionada()
                        tempo_soletrar_inicio = now + 0.6  # debounce de repetição
            else:
                tempo_soletrar_inicio = None
                ultima_letra_soletrada = ""

            # Banner Superior com a Palavra Formada
            card_s_w = 640
            card_s_h = 100
            cx_s = w // 2 - card_s_w // 2
            cy_s = 20
            desenhar_retangulo_arredondado(
                img, (cx_s, cy_s), (cx_s + card_s_w, cy_s + card_s_h),
                cor_fundo=(12, 14, 22), cor_borda=(0, 220, 255), raio=14, alpha=0.90, espessura_borda=2
            )
            cv2.putText(img, "SOLETRADOR EM LIBRAS (SOLETRE SEU NOME NO AR)", (cx_s + 20, cy_s + 28), cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 220, 255), 1, cv2.LINE_AA)

            palavra_display = palavra_soletrada if palavra_soletrada else "[ Segure a pose para adicionar letras... ]"
            cor_palavra = (0, 255, 140) if palavra_soletrada else (120, 125, 140)
            cv2.putText(img, palavra_display, (cx_s + 20, cy_s + 68), cv2.FONT_HERSHEY_DUPLEX, 0.95, cor_palavra, 2, cv2.LINE_AA)
            cv2.putText(img, "[BACKSPACE: Apagar letra] | [DELETE: Limpar palavra]", (cx_s + 20, cy_s + 90), cv2.FONT_HERSHEY_DUPLEX, 0.38, (160, 165, 180), 1, cv2.LINE_AA)

            # Pílula no Canto com a letra atual
            bx = w - 195
            by = 25
            desenhar_retangulo_arredondado(img, (bx, by), (bx + 170, by + 85), (12, 14, 22), (0, 220, 255), raio=12, alpha=0.88)
            scale_lbl = 1.8 if len(lbl_show) <= 2 else 1.05
            offset_lbl_y = 56 if len(lbl_show) <= 2 else 50
            cv2.putText(img, lbl_show, (bx + 15, by + offset_lbl_y), cv2.FONT_HERSHEY_DUPLEX, scale_lbl, cor_glow, 2, cv2.LINE_AA)
            offset_prec_x = bx + 90 if len(lbl_show) <= 2 else bx + 105
            cv2.putText(img, f"{confianca_val:.0f}%", (offset_prec_x, by + 52), cv2.FONT_HERSHEY_DUPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

            # Barra de progresso para adicionar letra
            if prog_soletrar > 0:
                cv2.rectangle(img, (bx + 15, by + 68), (bx + 15 + int(135 * prog_soletrar), by + 74), (0, 255, 140), -1)

        # AVISO QUANDO APENAS A MÃO ESQUERDA FOR DETECTADA
        if mao_esquerda_apenas and not mao_direita_encontrada:
            cx_av = w // 2 - 200
            cy_av = 130
            desenhar_retangulo_arredondado(
                img, (cx_av, cy_av), (cx_av + 400, cy_av + 58),
                cor_fundo=(35, 20, 10), cor_borda=(0, 160, 255), raio=12, alpha=0.92, espessura_borda=2
            )
            cv2.putText(
                img, "[ ! ] USE A MAO DIREITA", (cx_av + 32, cy_av + 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 200, 255), 2, cv2.LINE_AA
            )

        # BANNER DE CONFIRMAÇÃO DE CALIBRAÇÃO SALVA
        if time.time() < calibrado_feedback_timer:
            cx_c = w // 2 - 270
            cy_c = 75
            desenhar_retangulo_arredondado(
                img, (cx_c, cy_c), (cx_c + 540, cy_c + 56),
                cor_fundo=(10, 35, 20), cor_borda=(0, 255, 140), raio=12, alpha=0.94, espessura_borda=2
            )
            cv2.putText(
                img, calibrado_feedback_txt, (cx_c + 20, cy_c + 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 255, 140), 1, cv2.LINE_AA
            )

        # MODAL DE CALIBRAÇÃO (AO APERTAR A-Z ou 1/2)
        if modo_gravando:
            cx_m = w // 2 - 250
            cy_m = h // 2 - 75
            desenhar_retangulo_arredondado(
                img, (cx_m, cy_m), (cx_m + 500, cy_m + 150),
                cor_fundo=(10, 12, 18), cor_borda=(0, 255, 180), raio=14, alpha=0.92, espessura_borda=2
            )
            cv2.putText(
                img, f"CALIBRANDO: '{letra_alvo_gravacao}'",
                (cx_m + 35, cy_m + 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 180), 2, cv2.LINE_AA
            )
            cv2.putText(
                img, "Mantenha a MAO DIREITA estavel na camera...",
                (cx_m + 35, cy_m + 75), cv2.FONT_HERSHEY_DUPLEX, 0.48, (210, 215, 225), 1, cv2.LINE_AA
            )
            prog_g = frames_gravados / float(total_frames_gravar)
            w_prog_box = 430
            cv2.rectangle(img, (cx_m + 35, cy_m + 95), (cx_m + 35 + w_prog_box, cy_m + 112), (30, 34, 45), -1)
            cv2.rectangle(img, (cx_m + 35, cy_m + 95), (cx_m + 35 + int(w_prog_box * prog_g), cy_m + 112), (0, 220, 255), -1)

        # ---------------------------------------------------------------------
        # RODAPÉ COM ATALHOS E INDICAÇÃO DO MODO (ROI sem cópia de frame inteiro)
        # ---------------------------------------------------------------------
        foot_h = 38
        foot_roi = img[h - foot_h:h, 0:w]
        foot_bg = np.full(foot_roi.shape, (10, 12, 18), dtype=np.uint8)
        cv2.addWeighted(foot_bg, 0.88, foot_roi, 0.12, 0, foot_roi)

        # Tag do Modo Atual (Teclas F1, F2, F3)
        cv2.putText(
            img, f"[F1/F2/F3] MODO: {nomes_modos[modo_tela]}", (20, h - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 140), 1, cv2.LINE_AA
        )

        status_espelho = "LIGADO" if espelhar_video else "DESLIGADO"
        cor_espelho = (0, 255, 140) if espelhar_video else (0, 220, 255)
        cv2.putText(
            img, f"[ESPACO] Espelho: {status_espelho}", (375, h - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.44, cor_espelho, 1, cv2.LINE_AA
        )

        # Indicador de Mão Direita Ativa
        cor_badge_mao = (0, 255, 140) if mao_direita_encontrada else ((0, 160, 255) if mao_esquerda_apenas else (140, 145, 160))
        txt_badge_mao = "MAO DIREITA OK" if mao_direita_encontrada else ("USE MAO DIREITA" if mao_esquerda_apenas else "AGUARDANDO MAO")
        cv2.putText(
            img, f"[{txt_badge_mao}]", (585, h - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.44, cor_badge_mao, 1, cv2.LINE_AA
        )

        cv2.putText(
            img, "[A-Z: Calibrar] | [TAB: Tela Cheia] | [ESC: Sair]",
            (775, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.42, (180, 185, 200), 1, cv2.LINE_AA
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

        # Detecção de teclas especiais (F1, F2, F3, Delete) compatível com Windows e Linux
        is_f1 = (key_raw in [7340032, 0x700000, 65470]) or ((key_raw >> 16) == 0x70)
        is_f2 = (key_raw in [7405568, 0x710000, 65471]) or ((key_raw >> 16) == 0x71)
        is_f3 = (key_raw in [7471104, 0x720000, 65472]) or ((key_raw >> 16) == 0x72)
        is_delete = (key_raw in [3014656, 0x2E0000, 65535, 127]) or ((key_raw >> 16) == 0x2E)

        # Tratamento de Teclas
        if key == 27:  # ESC: Salva dataset antes de sair
            ml_engine.salvar_dataset()
            break
        elif is_f1:  # F1: Modo 0 (Tela Limpa)
            modo_tela = 0
            print(f"[MODO] Alterado para: {nomes_modos[modo_tela]}")
        elif is_f2:  # F2: Modo 1 (Jogo Desafio)
            modo_tela = 1
            print(f"[MODO] Alterado para: {nomes_modos[modo_tela]}")
        elif is_f3:  # F3: Modo 2 (Soletrador)
            modo_tela = 2
            print(f"[MODO] Alterado para: {nomes_modos[modo_tela]}")
        elif key == 32:  # ESPAÇO: Espelho
            espelhar_video = not espelhar_video
        elif key == 9:  # TAB: Tela Cheia
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == 8:  # Backspace: apaga letra no soletrador
            if modo_tela == 2 and palavra_soletrada:
                palavra_soletrada = palavra_soletrada[:-1]
        elif is_delete:  # Delete: limpa palavra no soletrador
            if modo_tela == 2:
                palavra_soletrada = ""
        elif key == ord('1'):  # Tecla 1: Calibra Joinha
            modo_gravando = True
            letra_alvo_gravacao = "JOINHA"
            frames_gravados = 0
            ml_engine.iniciar_gravacao_classe("JOINHA")
        elif key == ord('2'):  # Tecla 2: Calibra Censura
            modo_gravando = True
            letra_alvo_gravacao = "CENSURA"
            frames_gravados = 0
            ml_engine.iniciar_gravacao_classe("CENSURA")
        elif (65 <= key <= 90) or (97 <= key <= 122):
            # Todas as 26 letras (incluindo M e C) agora calibram sem conflito!
            char_digitado = chr(key).upper()
            modo_gravando = True
            letra_alvo_gravacao = char_digitado
            frames_gravados = 0
            ml_engine.iniciar_gravacao_classe(char_digitado)
            print(f"[CALIBRACAO] Gravando para a letra '{letra_alvo_gravacao}'...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
