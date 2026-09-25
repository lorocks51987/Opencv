"""
=============================================================================
     MATH BLITZ: DESAFIO DOS DEDOS - ADS UNIMAR ABERTA (EDIÇÃO STAND)
=============================================================================
Jogo arcade de agilidade mental e reflexos biométricos:
- Resolva continhas matemáticas e desafios rápidos mostrando a quantidade
  exata de dedos no ar com as duas mãos (0 a 10 dedos)!
- Sistema de Combos dinâmicos (x1 até x5) e multiplicador de pontuação.
- Cronômetro eletrizante de 45 segundos por partida.
- Tela de Game Over com placar limpo e salvamento do Maior Recorde do Stand.
- Otimização para 60 FPS estáveis com inferência reduzida (640x360).
- Efeitos visuais Cyber-Clean Glassmorphism e sons procedurais.
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

# Sons procedurais nativos do Windows
try:
    import winsound
    def tocar_som_acerto():
        winsound.Beep(988, 70)
        winsound.Beep(1318, 110)
    def tocar_som_tick():
        winsound.Beep(880, 40)
    def tocar_som_vitoria():
        threading.Thread(target=_worker_vitoria, daemon=True).start()
    def _worker_vitoria():
        for freq in [523, 659, 784, 1046]:
            winsound.Beep(freq, 60)
    def tocar_som_fim():
        threading.Thread(target=_worker_fim, daemon=True).start()
    def _worker_fim():
        for freq in [440, 370, 311]:
            winsound.Beep(freq, 90)
except Exception:
    def tocar_som_acerto(): pass
    def tocar_som_tick(): pass
    def tocar_som_vitoria(): pass
    def tocar_som_fim(): pass

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
# SISTEMA DE PARTÍCULAS / EXPLOSÃO DE ACERTO
# =============================================================================
class Particula:
    def __init__(self, x, y, cor=None):
        self.x = float(x)
        self.y = float(y)
        ang = random.uniform(0, 2 * math.pi)
        vel = random.uniform(4.0, 13.0)
        self.vx = math.cos(ang) * vel
        self.vy = math.sin(ang) * vel - random.uniform(2.0, 5.0)
        self.cor = cor or random.choice([(0, 255, 140), (0, 220, 255), (255, 200, 0), (255, 255, 255)])
        self.raio = random.randint(3, 6)
        self.vida = 1.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.38
        self.vida -= 0.04
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
                print(f"[OK] Câmera conectada no índice: {idx}")
                return cap
            cap.release()
    return None

# =============================================================================
# ANÁLISE ANATÔMICA DOS DEDOS (AMBAS AS MÃOS 0 A 10)
# =============================================================================
def analisar_dedos_mao(hand_landmarks, is_right_hand, w, h):
    """Retorna lista de 5 booleanos para cada dedo e coordenadas das pontas erguidas."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    escala = max(10.0, math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]))

    dedos_up = [False, False, False, False, False]
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    # Indicador, Médio, Anelar, Mindinho
    for i, (tip, pip) in enumerate(zip(tip_ids, pip_ids)):
        if pts[tip][1] < pts[pip][1] - (escala * 0.10):
            dedos_up[i + 1] = True

    # Polegar
    dist_p4_p17 = math.hypot(pts[4][0] - pts[17][0], pts[4][1] - pts[17][1]) / escala
    dist_p4_p2 = math.hypot(pts[4][0] - pts[2][0], pts[4][1] - pts[2][1]) / escala
    if dist_p4_p17 > 0.65 and dist_p4_p2 > 0.35:
        dedos_up[0] = True

    pontas = []
    ids_pontas = [4, 8, 12, 16, 20]
    for i, erguido in enumerate(dedos_up):
        if erguido:
            pontas.append(pts[ids_pontas[i]])

    return dedos_up, pontas, pts

# =============================================================================
# FLUXO PRINCIPAL DO MATH BLITZ
# =============================================================================
def main():
    print("Iniciando Math Blitz: Desafio dos Dedos (ADS Unimar Aberta)...")
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

    nome_janela = "Math Blitz: Desafio dos Dedos | ADS UNIMAR ABERTA"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    fullscreen = False
    espelhar_video = True

    # Estados do Jogo:
    # 0 = Tela Inicial (Pronto para começar)
    # 1 = Partida Ativa (45 segundos de jogo)
    # 2 = Game Over (Placar e Recorde)
    estado = 0

    duracao_jogo = 45.0
    tempo_inicio = 0.0
    score = 0
    combo = 1
    recorde = carregar_recorde()
    novo_recorde_batido = False

    pergunta_txt = ""
    resposta_alvo = 0
    tempo_resposta_sustentada = 0.0
    anim_acerto_timer = 0.0

    particulas = []
    fps_tempo = time.time()
    fps_cont = 0
    fps_display = 0

    def sortear_desafio():
        nonlocal pergunta_txt, resposta_alvo
        tipo = random.choice(["soma", "soma", "sub", "direto"])
        if tipo == "soma":
            a = random.randint(1, 5)
            b = random.randint(1, 5)
            resposta_alvo = a + b
            pergunta_txt = f"QUANTO E: {a} + {b} = ?"
        elif tipo == "sub":
            a = random.randint(4, 10)
            b = random.randint(1, a - 1)
            resposta_alvo = a - b
            pergunta_txt = f"QUANTO E: {a} - {b} = ?"
        else:
            resposta_alvo = random.randint(1, 10)
            pergunta_txt = f"MOSTRE EXATAMENTE: {resposta_alvo} DEDOS!"

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

        total_dedos = 0
        pontas_dedos = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label_raw = handedness_info.classification[0].label
                is_right_hand = (label_raw == "Right") if espelhar_video else (label_raw == "Left")

                dedos_up, pontas, pts_raw = analisar_dedos_mao(hand_landmarks, is_right_hand, w, h)
                total_dedos += sum(dedos_up)
                pontas_dedos.extend(pontas)

                # Esqueleto Biométrico Cyber
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
                for pt in pts_raw:
                    cv2.circle(img, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Atualiza partículas
        particulas = [p for p in particulas if p.update()]
        for p in particulas:
            p.draw(img)

        # ---------------------------------------------------------------------
        # ESTADO 0: TELA INICIAL (BOAS-VINDAS)
        # ---------------------------------------------------------------------
        if estado == 0:
            scrim = np.full(img.shape, (10, 12, 20), dtype=np.uint8)
            cv2.addWeighted(scrim, 0.75, img, 0.25, 0, img)

            cx = w // 2 - 340
            cy = h // 2 - 170
            desenhar_retangulo_arredondado(
                img, (cx, cy), (cx + 680, cy + 340),
                cor_fundo=(12, 14, 24), cor_borda=(0, 255, 140), raio=16, alpha=0.92, espessura_borda=2
            )

            cv2.putText(img, "ADS * UNIMAR ABERTA", (cx + 40, cy + 45),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "MATH BLITZ: DESAFIO DOS DEDOS", (cx + 40, cy + 85),
                        cv2.FONT_HERSHEY_DUPLEX, 0.88, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(img, (cx + 40, cy + 105), (cx + 640, cy + 105), (45, 55, 75), 1)

            cv2.putText(img, "Resolva continhas no ar mostrando a quantidade de dedos!", (cx + 40, cy + 150),
                        cv2.FONT_HERSHEY_DUPLEX, 0.50, (210, 215, 225), 1, cv2.LINE_AA)
            cv2.putText(img, "Use as duas maos simultaneamente (de 0 a 10 dedos)!", (cx + 40, cy + 185),
                        cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"Maior Recorde de Todo o Evento: {recorde} PONTOS", (cx + 40, cy + 230),
                        cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 200, 0), 1, cv2.LINE_AA)

            cv2.line(img, (cx + 40, cy + 258), (cx + 640, cy + 258), (45, 55, 75), 1)
            cv2.putText(img, "Pressione [ESPACO] ou [R] para Iniciar a Partida!", (cx + 40, cy + 298),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 140), 2, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # ESTADO 1: PARTIDA ATIVA (45 SEGUNDOS)
        # ---------------------------------------------------------------------
        elif estado == 1:
            tempo_decorrido = now - tempo_inicio
            tempo_restante = max(0.0, duracao_jogo - tempo_decorrido)

            # Efeito sonoro do timer nos últimos 5 segundos
            if tempo_restante <= 5.0 and int(tempo_restante) != int(tempo_restante + 0.05):
                tocar_som_tick()

            # Fim do Tempo -> Game Over
            if tempo_restante <= 0:
                estado = 2
                if score > recorde:
                    recorde = score
                    salvar_recorde(recorde)
                    novo_recorde_batido = True
                    tocar_som_vitoria()
                else:
                    novo_recorde_batido = False
                    tocar_som_fim()

            # Card Superior com Pergunta
            card_w = 640
            card_h = 145
            cx = w // 2 - card_w // 2
            cy = 20

            cor_borda = (0, 255, 140) if (now < anim_acerto_timer) else (0, 220, 255)
            desenhar_retangulo_arredondado(
                img, (cx, cy), (cx + card_w, cy + card_h),
                cor_fundo=(12, 14, 24), cor_borda=cor_borda, raio=14, alpha=0.92, espessura_borda=2
            )

            # Barra de Informações do Jogo
            cv2.putText(
                img, f"SCORE: {score}   |   COMBO: x{combo}   |   TEMPO: {int(tempo_restante)}s",
                (cx + 35, cy + 34), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 200, 0), 1, cv2.LINE_AA
            )

            # Pergunta Central
            cv2.putText(
                img, pergunta_txt, (cx + 35, cy + 88),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Feedback dos Dedos do Jogador
            cor_feedback = (0, 255, 140) if total_dedos == resposta_alvo else (0, 220, 255)
            cv2.putText(
                img, f"Voce esta mostrando: {total_dedos} dedos", (cx + 35, cy + 124),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, cor_feedback, 1, cv2.LINE_AA
            )

            # Barra de Progresso do Tempo Restante
            prog_t = tempo_restante / duracao_jogo
            w_prog = card_w - 70
            cv2.rectangle(img, (cx + 35, cy + card_h - 10), (cx + 35 + w_prog, cy + card_h - 5), (30, 35, 50), -1)
            cv2.rectangle(img, (cx + 35, cy + card_h - 10), (cx + 35 + int(w_prog * prog_t), cy + card_h - 5), (0, 255, 140), -1)

            # Validação da Resposta
            if total_dedos == resposta_alvo:
                if tempo_resposta_sustentada == 0.0:
                    tempo_resposta_sustentada = now
                elif now - tempo_resposta_sustentada >= 0.32:
                    # Acertou!
                    pontos = 100 * combo
                    score += pontos
                    combo = min(5, combo + 1)
                    anim_acerto_timer = now + 0.55
                    tocar_som_acerto()

                    # Partículas nas pontas dos dedos
                    for pt in pontas_dedos:
                        for _ in range(6):
                            particulas.append(Particula(pt[0], pt[1]))

                    sortear_desafio()
                    tempo_resposta_sustentada = 0.0
            else:
                tempo_resposta_sustentada = 0.0

        # ---------------------------------------------------------------------
        # ESTADO 2: GAME OVER (PLACAR E RECORDE DO EVENTO)
        # ---------------------------------------------------------------------
        elif estado == 2:
            scrim = np.full(img.shape, (10, 12, 20), dtype=np.uint8)
            cv2.addWeighted(scrim, 0.82, img, 0.18, 0, img)

            cx = w // 2 - 320
            cy = h // 2 - 150
            desenhar_retangulo_arredondado(
                img, (cx, cy), (cx + 640, cy + 300),
                cor_fundo=(12, 14, 24), cor_borda=(0, 255, 140), raio=16, alpha=0.94, espessura_borda=2
            )

            cv2.putText(img, "FIM DE JOGO!", (cx + 205, cy + 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 140), 2, cv2.LINE_AA)
            cv2.line(img, (cx + 40, cy + 70), (cx + 600, cy + 70), (45, 55, 75), 1)

            cv2.putText(img, f"SUA PONTUACAO FINAL: {score} PONTOS", (cx + 45, cy + 125),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

            if novo_recorde_batido:
                cv2.putText(img, "NOVO RECORDE DO STAND BATIDO! PARABENS!", (cx + 45, cy + 175),
                            cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 255, 140), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, f"MAIOR RECORDE DO EVENTO: {recorde} PONTOS", (cx + 45, cy + 175),
                            cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 200, 0), 1, cv2.LINE_AA)

            cv2.line(img, (cx + 40, cy + 215), (cx + 600, cy + 215), (45, 55, 75), 1)
            cv2.putText(img, "Pressione [ESPACO] ou [R] para Jogar Novamente!", (cx + 45, cy + 260),
                        cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 220, 255), 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # RODAPÉ CYBER-CLEAN PADRONIZADO (38px)
        # ---------------------------------------------------------------------
        foot_h = 38
        foot_roi = img[h - foot_h:h, 0:w]
        foot_bg = np.full(foot_roi.shape, (10, 12, 18), dtype=np.uint8)
        cv2.addWeighted(foot_bg, 0.88, foot_roi, 0.12, 0, foot_roi)

        cv2.putText(
            img, "ADS * UNIMAR ABERTA | MATH BLITZ", (20, h - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 140), 1, cv2.LINE_AA
        )

        status_espelho = "LIGADO" if espelhar_video else "DESLIGADO"
        cor_espelho = (0, 255, 140) if espelhar_video else (0, 220, 255)
        cv2.putText(
            img, f"[ESPACO] Espelho: {status_espelho}", (365, h - 14),
            cv2.FONT_HERSHEY_DUPLEX, 0.44, cor_espelho, 1, cv2.LINE_AA
        )

        cv2.putText(
            img, "[R: Reiniciar] | [TAB/F: Tela Cheia] | [ESC: Sair]",
            (590, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.44, (180, 185, 200), 1, cv2.LINE_AA
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
        key = cv2.waitKey(1) & 0xFF

        # Tratamento de Teclas
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC / Q
            break
        elif key == 32:  # ESPAÇO: Ação contextual ou Espelho
            if estado == 0 or estado == 2:
                estado = 1
                score = 0
                combo = 1
                tempo_inicio = time.time()
                sortear_desafio()
            else:
                espelhar_video = not espelhar_video
        elif key == ord('r') or key == ord('R'):  # R: Reiniciar partida
            estado = 1
            score = 0
            combo = 1
            tempo_inicio = time.time()
            sortear_desafio()
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
