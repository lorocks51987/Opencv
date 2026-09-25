"""
=============================================================================
     VIRTUAL PAINTER - ADS UNIMAR (COM CAPTURA E ENVIO POR E-MAIL)
=============================================================================
Recursos da versão:
- Tela de Cadastro Inicial: Coleta Nome e E-mail do aluno antes de pintar.
- Pintura no ar com OpenCV & MediaPipe (1 dedo = desenha, 2 dedos = seleciona cor).
- Moldura Oficial com tema "ADS • UNIMAR ABERTA" aplicada sobre a arte.
- Exportação em alta qualidade para 'galeria_visitantes/' e planilha 'leads_visitantes.csv'.
- Disparo de E-mail automático (em thread assíncrona, sem travar o vídeo).
- Suporte a fila offline se não houver internet no momento do evento.
- Botão "Limpar Tela", "Finalizar e Enviar" e suporte a tela cheia (F).
=============================================================================
"""

import cv2
import numpy as np
import os
import sys
import time
import json
import csv
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Carrega módulo de rastreamento de mãos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import HandTrackingModule as htm

# Pastas de dados
FOLDER_SAVED = os.path.join(SCRIPT_DIR, "galeria_visitantes")
FILE_CSV = os.path.join(SCRIPT_DIR, "leads_visitantes.csv")
FILE_CONFIG = os.path.join(SCRIPT_DIR, "config_email.json")
FILE_QUEUE = os.path.join(SCRIPT_DIR, "fila_emails.json")
os.makedirs(FOLDER_SAVED, exist_ok=True)

# Paleta de Cores (Formato BGR)
CORES = [
    {"nome": "VERDE NEON",   "bgr": (0, 255, 120),  "rgb_ui": (0, 255, 120)},
    {"nome": "ROSA CHOQUE",  "bgr": (180, 50, 255), "rgb_ui": (180, 50, 255)},
    {"nome": "AZUL CIANO",   "bgr": (255, 200, 0),   "rgb_ui": (255, 200, 0)},
    {"nome": "AMARELO OURO", "bgr": (0, 230, 255),   "rgb_ui": (0, 230, 255)},
    {"nome": "BRANCO NEVE",  "bgr": (255, 255, 255), "rgb_ui": (255, 255, 255)},
    {"nome": "BORRACHA",     "bgr": (0, 0, 0),       "rgb_ui": (80, 80, 80)}
]

# =============================================================================
# MOTOR DE ENVIO DE E-MAIL (ASSÍNCRONO)
# =============================================================================
def enviar_email_async(destinatario, nome_aluno, caminho_imagem):
    threading.Thread(
        target=_worker_envio_email,
        args=(destinatario, nome_aluno, caminho_imagem),
        daemon=True
    ).start()

def _worker_envio_email(destinatario, nome_aluno, caminho_imagem):
    if not os.path.exists(FILE_CONFIG):
        return

    try:
        with open(FILE_CONFIG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[ERRO CONFIG] {e}")
        return

    # Se o envio real não estiver ativado ou e-mail for inválido
    if not cfg.get("habilitar_envio_real", False) or "@" not in destinatario:
        print(f"[FILA] E-mail para {destinatario} gravado na fila (envio real desativado no config_email.json).")
        _salvar_fila_email(destinatario, nome_aluno, caminho_imagem)
        return

    try:
        remetente = cfg.get("email_remetente")
        senha = cfg.get("senha_app")
        servidor_smtp = cfg.get("servidor_smtp", "smtp.gmail.com")
        porta = cfg.get("porta_smtp", 587)
        assunto = cfg.get("assunto", "Sua Arte na Unimar Aberta! 🎨")
        corpo_template = cfg.get("mensagem_corpo", "Ola {nome}!\n\nAqui esta a sua arte criada no stand de ADS!")
        corpo = corpo_template.format(nome=nome_aluno)

        msg = MIMEMultipart()
        msg["From"] = remetente
        msg["To"] = destinatario
        msg["Subject"] = assunto
        msg.attach(MIMEText(corpo, "plain", "utf-8"))

        if os.path.exists(caminho_imagem):
            with open(caminho_imagem, "rb") as img_f:
                anexo = MIMEImage(img_f.read(), name=os.path.basename(caminho_imagem))
                msg.attach(anexo)

        server = smtplib.SMTP(servidor_smtp, porta, timeout=10)
        server.starttls()
        server.login(remetente, senha)
        server.sendmail(remetente, [destinatario], msg.as_string())
        server.quit()
        print(f"[SUCESSO] E-mail enviado para {destinatario}!")
    except Exception as e:
        print(f"[FALHA ENVIO] {e}. Salvando na fila para reenvio posterior.")
        _salvar_fila_email(destinatario, nome_aluno, caminho_imagem)

def _salvar_fila_email(destinatario, nome, img_path):
    fila = []
    if os.path.exists(FILE_QUEUE):
        try:
            with open(FILE_QUEUE, "r", encoding="utf-8") as f:
                fila = json.load(f)
        except Exception:
            fila = []
    fila.append({
        "destinatario": destinatario,
        "nome": nome,
        "caminho_imagem": img_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    try:
        with open(FILE_QUEUE, "w", encoding="utf-8") as f:
            json.dump(fila, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERRO FILA] {e}")

def registrar_lead_csv(nome, email, arquivo):
    novo = not os.path.exists(FILE_CSV)
    try:
        with open(FILE_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if novo:
                writer.writerow(["Data", "Hora", "Nome", "Email", "Arquivo"])
            agora = time.localtime()
            data_str = time.strftime("%d/%m/%Y", agora)
            hora_str = time.strftime("%H:%M:%S", agora)
            writer.writerow([data_str, hora_str, nome, email, arquivo])
    except Exception as e:
        print(f"[ERRO CSV] {e}")

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
# BUSCA DE CÂMERA
# =============================================================================
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
# GERAÇÃO DA MOLDURA OFICIAL DE ADS UNIMAR
# =============================================================================
def gerar_moldura_oficial(imgCanvas, nome_aluno, email_aluno):
    h, w = imgCanvas.shape[:2]

    # Cria fundo temático escuro
    moldura = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(12 + (y / h) * 16)
        moldura[y, :] = (val + 5, val, val + 15)

    # Copia a pintura do aluno com mesclagem sobre o fundo
    gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    fundo_recortado = cv2.bitwise_and(moldura, moldura, mask=mask_inv)
    arte_recortada = cv2.bitwise_and(imgCanvas, imgCanvas, mask=mask)
    resultado = cv2.add(fundo_recortado, arte_recortada)

    # Moldura Superior (Header)
    header_bar = 70
    overlay = resultado.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_bar), (10, 10, 20), -1)
    resultado = cv2.addWeighted(overlay, 0.85, resultado, 0.15, 0)
    cv2.line(resultado, (0, header_bar), (w, header_bar), (0, 220, 255), 3)

    cv2.putText(
        resultado, "ANALISE E DESENVOLVIMENTO DE SISTEMAS * UNIMAR ABERTA",
        (30, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        resultado, "OBRA DE ARTE COM VISAO COMPUTACIONAL (OPENCV & IA)",
        (30, 58), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Moldura Inferior (Footer)
    footer_bar = 55
    overlay_f = resultado.copy()
    cv2.rectangle(overlay_f, (0, h - footer_bar), (w, h), (10, 10, 20), -1)
    resultado = cv2.addWeighted(overlay_f, 0.85, resultado, 0.15, 0)
    cv2.line(resultado, (0, h - footer_bar), (w, h - footer_bar), (0, 220, 255), 2)

    agora_txt = time.strftime("%d/%m/%Y as %H:%M")
    autor_txt = f"Artista: {nome_aluno} ({email_aluno})"
    info_txt = f"Criado em: {agora_txt} * Unimar"

    cv2.putText(
        resultado, autor_txt, (30, h - 22),
        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 120), 1, cv2.LINE_AA
    )
    cv2.putText(
        resultado, info_txt, (w - 380, h - 22),
        cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
    )

    # Borda externa geral
    cv2.rectangle(resultado, (0, 0), (w - 1, h - 1), (0, 220, 255), 3)

    return resultado

# =============================================================================
# FLUXO PRINCIPAL DO VIRTUAL PAINTER
# =============================================================================
def main():
    print("Iniciando Virtual Painter com Envio por E-mail (ADS Unimar)...")
    cap = encontrar_camera()
    if cap is None:
        print("[ERRO] Nenhuma câmera encontrada!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = htm.handDetector(detectionCon=0.8, trackCon=0.7)
    nome_janela = "Virtual Painter com E-mail | ADS UNIMAR ABERTA"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    fullscreen = False
    espelhar_video = True

    # Estados
    # 0 = Tela de Cadastro (Nome e Email)
    # 1 = Tela de Pintura
    # 2 = Tela de Sucesso / Envio
    estado = 0

    input_nome = ""
    input_email = ""
    campo_ativo = 0  # 0 = Nome, 1 = Email

    # Variáveis da Pintura
    cor_atual_idx = 0
    drawColor = CORES[cor_atual_idx]["bgr"]
    brushThickness = 12
    eraseThickness = 65
    xp, yp = 0, 0
    imgCanvas = None

    msg_status = "Escolha as cores no topo com 2 dedos e desenhe com 1 dedo!"
    msg_timer = time.time() + 4.0
    timer_sucesso = 0
    caminho_salvo_recente = ""

    fps_tempo = time.time()
    fps_cont = 0
    fps_display = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        if espelhar_video:
            img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        # Inicializa ou redimensiona Canvas conforme a resolução real da câmera
        if imgCanvas is None:
            imgCanvas = np.zeros((h, w, 3), np.uint8)
        elif imgCanvas.shape[:2] != (h, w):
            imgCanvas = cv2.resize(imgCanvas, (w, h))

        # ---------------------------------------------------------------------
        # ESTADO 0: TELA DE CADASTRO DO ALUNO (CYBER-CLEAN GLASSMORPHISM)
        # ---------------------------------------------------------------------
        if estado == 0:
            scrim = np.full(img.shape, (10, 12, 20), dtype=np.uint8)
            cv2.addWeighted(scrim, 0.78, img, 0.22, 0, img)

            # Card Central Glassmorphism
            card_w = 700
            card_h = 420
            cx = w // 2 - card_w // 2
            cy = h // 2 - card_h // 2

            desenhar_retangulo_arredondado(
                img, (cx, cy), (cx + card_w, cy + card_h),
                cor_fundo=(12, 14, 24), cor_borda=(0, 220, 255), raio=16, alpha=0.92, espessura_borda=2
            )

            # Cabeçalho do Card
            cv2.putText(
                img, "ADS * UNIMAR ABERTA", (cx + 40, cy + 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "PINTURA VIRTUAL COM IA & OPENCV", (cx + 40, cy + 78),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                img, "Desenhe no ar com as maos e receba sua obra no seu e-mail!", (cx + 40, cy + 108),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 140), 1, cv2.LINE_AA
            )
            cv2.line(img, (cx + 40, cy + 122), (cx + card_w - 40, cy + 122), (45, 50, 70), 1)

            # Campo 1: NOME
            box1_y = cy + 148
            border1 = (0, 255, 140) if campo_ativo == 0 else (60, 65, 80)
            desenhar_retangulo_arredondado(
                img, (cx + 40, box1_y), (cx + card_w - 40, box1_y + 48),
                cor_fundo=(16, 20, 32), cor_borda=border1, raio=10, alpha=0.90, espessura_borda=2 if campo_ativo == 0 else 1
            )
            cv2.putText(
                img, "SEU NOME:", (cx + 45, box1_y - 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA
            )
            nome_display = input_nome if input_nome else "Clique aqui e digite seu nome..."
            cor_txt1 = (255, 255, 255) if input_nome else (130, 130, 130)
            cursor1 = "|" if (campo_ativo == 0 and int(time.time() * 2) % 2 == 0) else ""
            cv2.putText(
                img, nome_display + cursor1, (cx + 55, box1_y + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, cor_txt1, 1, cv2.LINE_AA
            )

            # Campo 2: E-MAIL
            box2_y = cy + 238
            border2 = (0, 255, 140) if campo_ativo == 1 else (60, 65, 80)
            desenhar_retangulo_arredondado(
                img, (cx + 40, box2_y), (cx + card_w - 40, box2_y + 48),
                cor_fundo=(16, 20, 32), cor_borda=border2, raio=10, alpha=0.90, espessura_borda=2 if campo_ativo == 1 else 1
            )
            cv2.putText(
                img, "SEU E-MAIL (PARA RECEBER A ARTE):", (cx + 45, box2_y - 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA
            )
            email_display = input_email if input_email else "exemplo@gmail.com..."
            cor_txt2 = (255, 255, 255) if input_email else (130, 130, 130)
            cursor2 = "|" if (campo_ativo == 1 and int(time.time() * 2) % 2 == 0) else ""
            cv2.putText(
                img, email_display + cursor2, (cx + 55, box2_y + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, cor_txt2, 1, cv2.LINE_AA
            )

            # Botão Começar / Instruções
            cv2.putText(
                img, "[TAB]: Trocar de Campo  |  [ENTER]: Comecar a Pintar",
                (cx + 95, cy + 335), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 220, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "Ou pressione [ESPACO] para pintar sem cadastro",
                (cx + 130, cy + 375), cv2.FONT_HERSHEY_DUPLEX, 0.48, (160, 165, 180), 1, cv2.LINE_AA
            )

        # ---------------------------------------------------------------------
        # ESTADO 1: TELA DE PINTURA VIRTUAL
        # ---------------------------------------------------------------------
        elif estado == 1:
            img = detector.findHands(img, draw=False)
            lmList, _ = detector.findPosition(img, draw=False)

            num_botoes = len(CORES) + 2  # 6 Cores + Limpar + Enviar
            btn_w = w // num_botoes
            header_h = 76
            modo = "NEUTRO"

            if len(lmList) == 21:
                x1, y1 = lmList[8][1:]   # Indicador
                x2, y2 = lmList[12][1:]  # Médio
                fingers = detector.fingersUp()

                # MODO SELEÇÃO: Indicador + Médio levantados
                if fingers[1] and fingers[2]:
                    modo = "SELECAO"
                    xp, yp = 0, 0

                    if y1 < header_h:
                        btn_idx = x1 // btn_w
                        if btn_idx < len(CORES):
                            cor_atual_idx = btn_idx
                            drawColor = CORES[cor_atual_idx]["bgr"]
                            msg_status = f"Cor Selecionada: {CORES[cor_atual_idx]['nome']}"
                            msg_timer = time.time() + 2.0
                        elif btn_idx == len(CORES):
                            # Limpar
                            imgCanvas = np.zeros((h, w, 3), np.uint8)
                            msg_status = "Canvas Limpo!"
                            msg_timer = time.time() + 2.0
                        elif btn_idx == len(CORES) + 1:
                            # Finalizar e Enviar
                            estado = 2
                            timer_sucesso = time.time() + 4.0

                    cv2.circle(img, (x1, y1), 8, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, (x2, y2), 8, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 220, 255), 3, cv2.LINE_AA)

                # MODO DESENHO: Apenas indicador levantado
                elif fingers[1] and not fingers[2]:
                    modo = "DESENHANDO"
                    cv2.circle(img, (x1, y1), brushThickness // 2 + 5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(img, (x1, y1), brushThickness // 2 + 2, drawColor, -1, cv2.LINE_AA)

                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    thick = eraseThickness if drawColor == (0, 0, 0) else brushThickness
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thick, cv2.LINE_AA)
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0
            else:
                xp, yp = 0, 0

            # Mesclagem do Canvas com a imagem da câmera
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)

            # CABEÇALHO CYBER-CLEAN (ROI sem cópia de frame inteiro)
            head_roi = img[0:header_h, 0:w]
            head_bg = np.full(head_roi.shape, (12, 14, 22), dtype=np.uint8)
            cv2.addWeighted(head_bg, 0.85, head_roi, 0.15, 0, head_roi)
            cv2.line(img, (0, header_h), (w, header_h), (0, 220, 255), 2, cv2.LINE_AA)

            # Pílulas de Cores Arredondadas
            for i, c in enumerate(CORES):
                bx = i * btn_w
                is_sel = (i == cor_atual_idx)
                cor_bg = c["bgr"] if c["bgr"] != (0, 0, 0) else (32, 35, 45)
                borda_col = (255, 255, 255) if is_sel else (60, 65, 80)

                desenhar_retangulo_arredondado(
                    img, (bx + 4, 8), (bx + btn_w - 4, header_h - 8),
                    cor_fundo=cor_bg, cor_borda=borda_col, raio=10, alpha=0.92, espessura_borda=3 if is_sel else 1
                )

                txt_col = (0, 0, 0) if (c["bgr"] != (0, 0, 0) and is_sel) else (255, 255, 255)
                cv2.putText(
                    img, c["nome"], (bx + 8, header_h // 2 + 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, txt_col, 1, cv2.LINE_AA
                )
                if is_sel:
                    cv2.circle(img, (bx + btn_w // 2, 14), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # Botão Limpar (C)
            bx_clean = len(CORES) * btn_w
            desenhar_retangulo_arredondado(
                img, (bx_clean + 4, 8), (bx_clean + btn_w - 4, header_h - 8),
                cor_fundo=(40, 20, 30), cor_borda=(0, 100, 255), raio=10, alpha=0.92, espessura_borda=2
            )
            cv2.putText(
                img, "LIMPAR (C)", (bx_clean + 10, header_h // 2 + 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
            )

            # Botão Finalizar e Enviar (ENTER)
            bx_send = (len(CORES) + 1) * btn_w
            desenhar_retangulo_arredondado(
                img, (bx_send + 4, 8), (w - 6, header_h - 8),
                cor_fundo=(10, 45, 25), cor_borda=(0, 255, 140), raio=10, alpha=0.92, espessura_borda=2
            )
            cv2.putText(
                img, "ENVIAR (ENTER)", (bx_send + 10, header_h // 2 + 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )

            # RODAPÉ CYBER-CLEAN PADRONIZADO (38px)
            foot_h = 38
            foot_roi = img[h - foot_h:h, 0:w]
            foot_bg = np.full(foot_roi.shape, (10, 12, 18), dtype=np.uint8)
            cv2.addWeighted(foot_bg, 0.88, foot_roi, 0.12, 0, foot_roi)

            status_espelho = "LIGADO" if espelhar_video else "DESLIGADO"
            cor_espelho = (0, 255, 140) if espelhar_video else (0, 220, 255)

            cv2.putText(
                img, "ADS * UNIMAR ABERTA | PINTOR VIRTUAL", (20, h - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 140), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, f"[ESPACO] Espelho: {status_espelho}", (365, h - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, cor_espelho, 1, cv2.LINE_AA
            )
            cv2.putText(
                img, f"Artista: {input_nome or 'Visitante'}", (585, h - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "[1 Dedo: Pintar] | [2 Dedos: Paleta] | [C: Limpar] | [ENTER: Enviar]",
                (770, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.40, (180, 185, 200), 1, cv2.LINE_AA
            )

        # ---------------------------------------------------------------------
        # ESTADO 2: FINALIZAÇÃO, MOLDURA E ENVIO
        # ---------------------------------------------------------------------
        elif estado == 2:
            if caminho_salvo_recente == "":
                nome_aluno_final = input_nome.strip() or "Visitante ADS"
                email_aluno_final = input_email.strip() or "visitante@unimar.br"

                timestamp_arq = time.strftime("%Y%m%d_%H%M%S")
                nome_sanitizado = "".join(c for c in nome_aluno_final if c.isalnum() or c in " _-")[:20]
                nome_arq = f"arte_{nome_sanitizado}_{timestamp_arq}.png"
                caminho_salvo_recente = os.path.join(FOLDER_SAVED, nome_arq)

                # Gera a moldura
                arte_final = gerar_moldura_oficial(imgCanvas, nome_aluno_final, email_aluno_final)
                cv2.imwrite(caminho_salvo_recente, arte_final)

                # Registra Lead no CSV
                registrar_lead_csv(nome_aluno_final, email_aluno_final, nome_arq)

                # Dispara o envio por e-mail em background
                enviar_email_async(email_aluno_final, nome_aluno_final, caminho_salvo_recente)

            # Scrim escuro
            scrim = np.full(img.shape, (10, 12, 18), dtype=np.uint8)
            cv2.addWeighted(scrim, 0.85, img, 0.15, 0, img)

            # Card Central Glassmorphism
            card_w = 640
            card_h = 330
            cx = w // 2 - card_w // 2
            cy = h // 2 - card_h // 2

            desenhar_retangulo_arredondado(
                img, (cx, cy), (cx + card_w, cy + card_h),
                cor_fundo=(12, 16, 26), cor_borda=(0, 255, 140), raio=16, alpha=0.94, espessura_borda=2
            )

            cv2.putText(
                img, "ARTE CRIADA COM SUCESSO!", (cx + 40, cy + 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.90, (0, 255, 140), 2, cv2.LINE_AA
            )
            cv2.line(img, (cx + 35, cy + 78), (cx + card_w - 35, cy + 78), (45, 55, 75), 1)

            cv2.putText(
                img, f"Artista: {input_nome or 'Visitante'}", (cx + 40, cy + 120),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, f"E-mail: {input_email or 'Nao informado'}", (cx + 40, cy + 160),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 220, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "Sua obra foi salva com a moldura oficial de ADS!", (cx + 40, cy + 205),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (200, 205, 215), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "Obrigado por visitar o stand de ADS da Unimar!", (cx + 40, cy + 245),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 140), 1, cv2.LINE_AA
            )

            # Barra de progresso para próximo aluno
            rem_t = max(0.0, timer_sucesso - time.time())
            prog_t = 1.0 - (rem_t / 4.0)
            cv2.rectangle(img, (cx + 40, cy + 280), (cx + card_w - 40, cy + 292), (25, 30, 42), -1)
            cv2.rectangle(img, (cx + 40, cy + 280), (cx + 40 + int((card_w - 80) * prog_t), cy + 292), (0, 255, 140), -1)

            # Retorno automático após 4 segundos para o próximo aluno
            if time.time() > timer_sucesso:
                estado = 0
                input_nome = ""
                input_email = ""
                campo_ativo = 0
                caminho_salvo_recente = ""
                imgCanvas = np.zeros((h, w, 3), np.uint8)

        # Contador de FPS
        fps_cont += 1
        if time.time() - fps_tempo >= 1.0:
            fps_display = fps_cont
            fps_cont = 0
            fps_tempo = time.time()

        if estado == 1:
            cv2.putText(
                img, f"{fps_display} FPS", (w - 85, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 140), 1, cv2.LINE_AA
            )

        cv2.imshow(nome_janela, img)
        key = cv2.waitKey(1) & 0xFF

        # Tratamento de Teclas
        if key == 27:  # ESC
            if estado == 1 or estado == 2:
                estado = 0
            else:
                break
        elif key == 9 or key == ord('f') or key == ord('F'):  # TAB / F
            if estado == 0 and key == 9:
                campo_ativo = 1 - campo_ativo
            else:
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        # Estado 0: Digitação de Cadastro
        if estado == 0:
            if key in [13, 10]:  # ENTER
                estado = 1
                imgCanvas = np.zeros((h, w, 3), np.uint8)
            elif key == 32:  # ESPAÇO
                if not input_nome and not input_email:
                    estado = 1
                    imgCanvas = np.zeros((h, w, 3), np.uint8)
                else:
                    if campo_ativo == 0:
                        input_nome += " "
                    else:
                        input_email += " "
            elif key == 8:  # Backspace
                if campo_ativo == 0:
                    input_nome = input_nome[:-1]
                else:
                    input_email = input_email[:-1]
            elif 32 < key <= 126:
                char = chr(key)
                if campo_ativo == 0:
                    if len(input_nome) < 28:
                        input_nome += char
                else:
                    if len(input_email) < 35:
                        input_email += char

        # Estado 1: Pintura
        elif estado == 1:
            if key == 32:  # ESPAÇO: Alternar espelhamento da câmera
                espelhar_video = not espelhar_video
            elif key == ord('c') or key == ord('C'):  # C: Limpar
                imgCanvas = np.zeros((h, w, 3), np.uint8)
                msg_status = "Canvas Limpo!"
                msg_timer = time.time() + 2.0
            elif key in [13, 10]:  # ENTER: Enviar
                estado = 2
                timer_sucesso = time.time() + 4.0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
