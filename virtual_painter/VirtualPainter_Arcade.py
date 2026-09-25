"""
=============================================================================
     VIRTUAL PAINTER - ADS UNIMAR (EDIÇÃO STAND COM QR CODE INSTANTÂNEO)
=============================================================================
Recursos da versão:
- Tela de Entrada Express: Cadastro rápido apenas com Nome/Apelido do visitante.
- Pintura no ar com OpenCV & MediaPipe (1 dedo = desenha, 2 dedos = seleciona cor).
- Moldura Oficial de Alta Resolução "ADS • UNIMAR ABERTA" aplicada sobre a arte.
- QR Code Instantâneo na Tela Final gerado nativamente via OpenCV:
  1. Nuvem (4G/5G/Wi-Fi): Upload rápido e seguro para download direto no smartphone.
  2. Rede Local: Micro-servidor HTTP embutido para download direto na rede do stand.
- O aluno aponta a câmera do celular para o QR Code e salva a foto na hora!
- 60 FPS estável, atalhos padronizados (F / TAB / ESPAÇO / C / ENTER / ESC).
=============================================================================
"""

import cv2
import numpy as np
import os
import sys
import time
import csv
import socket
import threading
import urllib.request
import json
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

# Carrega módulo de rastreamento de mãos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import HandTrackingModule as htm

# Pastas de dados
FOLDER_SAVED = os.path.join(SCRIPT_DIR, "galeria_visitantes")
FILE_CSV = os.path.join(SCRIPT_DIR, "leads_visitantes.csv")
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
# MICRO SERVIDOR HTTP LOCAL PARA DOWNLOAD NA REDE DO STAND
# =============================================================================
PORTA_LOCAL = 8080

def obter_ip_local():
    """Detecta o IP local da máquina na rede Wi-Fi/Ethernet."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class StandHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FOLDER_SAVED, **kwargs)

    def log_message(self, format, *args):
        # Silencia logs de requisição no console para manter terminal limpo
        pass

def iniciar_servidor_local():
    """Inicia servidor HTTP em background servindo a pasta da galeria."""
    global PORTA_LOCAL
    for porta in [8080, 8000, 8888, 9000]:
        try:
            httpd = TCPServer(("", porta), StandHTTPHandler)
            PORTA_LOCAL = porta
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            print(f"[SERVIDOR LOCAL] Galeria online em http://{obter_ip_local()}:{PORTA_LOCAL}/")
            return
        except OSError:
            continue
    print("[SERVIDOR LOCAL] Nenhuma porta disponível para o servidor local.")

# Inicia o micro servidor web silencioso em background
iniciar_servidor_local()

# =============================================================================
# GERADOR DE QR CODE E UPLOADER EM NUVEM
# =============================================================================
_qr_encoder = cv2.QRCodeEncoder_create()

def gerar_imagem_qr_code(url, tamanho=220):
    """Gera matriz BGR do QR Code em alta definição com borda de leitura branca."""
    try:
        qr = _qr_encoder.encode(url)
        if qr is None or qr.size == 0:
            return None
        # Redimensiona com interpolação vizinho mais próximo para módulos 100% nítidos
        qr_redim = cv2.resize(qr, (tamanho, tamanho), interpolation=cv2.INTER_NEAREST)
        # Borda silenciosa branca essencial para escaneamento rápido de câmeras mobile
        borda = 14
        qr_com_borda = cv2.copyMakeBorder(
            qr_redim, borda, borda, borda, borda, cv2.BORDER_CONSTANT, value=255
        )
        return cv2.cvtColor(qr_com_borda, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"[ERRO QR CODE] {e}")
        return None

def upload_nuvem_async(caminho_imagem, callback_sucesso):
    """Faz upload da imagem para um endpoint seguro e gratuito de download direto."""
    def _worker():
        try:
            url_api = "https://tmpfiles.org/api/v1/upload"
            boundary = "----WebKitFormBoundaryUnimarStand"
            nome_arq = os.path.basename(caminho_imagem)
            
            with open(caminho_imagem, "rb") as f:
                img_bytes = f.read()

            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{nome_arq}"\r\n'
                f"Content-Type: image/png\r\n\r\n"
            ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

            req = urllib.request.Request(
                url_api,
                data=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "ADS-Unimar-Stand"
                }
            )
            with urllib.request.urlopen(req, timeout=4.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                orig_url = data.get("data", {}).get("url", "")
                if orig_url:
                    # Formata URL de download direto para abrir imediatamente no navegador do celular
                    dl_url = orig_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    callback_sucesso(dl_url)
        except Exception:
            # Em caso de falha de conexão (offline), mantém o link local já configurado
            pass

    threading.Thread(target=_worker, daemon=True).start()

def registrar_lead_csv(nome, arquivo):
    """Registra a participação do visitante na planilha para controle do curso."""
    novo = not os.path.exists(FILE_CSV)
    try:
        with open(FILE_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if novo:
                writer.writerow(["Data", "Hora", "Nome", "Arquivo"])
            agora = time.localtime()
            data_str = time.strftime("%d/%m/%Y", agora)
            hora_str = time.strftime("%H:%M:%S", agora)
            writer.writerow([data_str, hora_str, nome, arquivo])
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
def gerar_moldura_oficial(imgCanvas, nome_aluno):
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
    autor_txt = f"Artista: {nome_aluno}"
    info_txt = f"Criado em: {agora_txt} * Unimar"

    cv2.putText(
        resultado, autor_txt, (30, h - 22),
        cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 140), 1, cv2.LINE_AA
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
    print("Iniciando Virtual Painter com QR Code Instantâneo (ADS Unimar)...")
    cap = encontrar_camera()
    if cap is None:
        print("[ERRO] Nenhuma câmera compatível detectada.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = htm.handDetector(detectionCon=0.8, trackCon=0.7)
    nome_janela = "Virtual Painter com QR Code | ADS UNIMAR ABERTA"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    fullscreen = False
    espelhar_video = True

    # Estados
    # 0 = Tela de Entrada / Nome
    # 1 = Tela de Pintura
    # 2 = Tela de Sucesso / QR Code
    estado = 0

    input_nome = ""

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

    # QR Code State 2
    qr_img_atual = None
    url_download_atual = ""
    status_qr_txt = "Gerando QR Code..."

    def atualizar_url_online(nova_url):
        nonlocal qr_img_atual, url_download_atual, status_qr_txt
        url_download_atual = nova_url
        qr_novo = gerar_imagem_qr_code(nova_url, tamanho=220)
        if qr_novo is not None:
            qr_img_atual = qr_novo
            status_qr_txt = "Link Online Pronto (Download Direto)"

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
        # ESTADO 0: TELA DE ENTRADA / NOME (CYBER-CLEAN GLASSMORPHISM)
        # ---------------------------------------------------------------------
        if estado == 0:
            scrim = np.full(img.shape, (10, 12, 20), dtype=np.uint8)
            cv2.addWeighted(scrim, 0.78, img, 0.22, 0, img)

            # Card Central Glassmorphism
            card_w = 720
            card_h = 360
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
                img, "Desenhe no ar com as maos e baixe sua arte direto no celular via QR Code!",
                (cx + 40, cy + 108), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 140), 1, cv2.LINE_AA
            )
            cv2.line(img, (cx + 40, cy + 124), (cx + card_w - 40, cy + 124), (45, 50, 70), 1)

            # Campo Único: NOME
            box1_y = cy + 155
            desenhar_retangulo_arredondado(
                img, (cx + 40, box1_y), (cx + card_w - 40, box1_y + 54),
                cor_fundo=(16, 20, 32), cor_borda=(0, 255, 140), raio=10, alpha=0.90, espessura_borda=2
            )
            cv2.putText(
                img, "SEU NOME / APELIDO:", (cx + 45, box1_y - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA
            )
            nome_disp = input_nome if input_nome else "Digite aqui ou pressione ESPACO para entrar direto..."
            cor_txt = (255, 255, 255) if input_nome else (100, 110, 130)
            cursor = "|" if (int(time.time() * 2) % 2 == 0) else ""
            cv2.putText(
                img, f"{nome_disp}{cursor}", (cx + 55, box1_y + 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, cor_txt, 1, cv2.LINE_AA
            )

            # Instruções e Atalhos no Rodapé do Card
            cv2.line(img, (cx + 40, cy + 245), (cx + card_w - 40, cy + 245), (45, 50, 70), 1)
            cv2.putText(
                img, "[ENTER] Iniciar com este Nome   |   [ESPACO] Entrar como Visitante Direto",
                (cx + 45, cy + 278), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 140), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "Dica: Ao terminar sua pintura, aponte a camera do celular para levar a foto!",
                (cx + 45, cy + 315), cv2.FONT_HERSHEY_DUPLEX, 0.44, (180, 185, 200), 1, cv2.LINE_AA
            )

        # ---------------------------------------------------------------------
        # ESTADO 1: TELA DE PINTURA (CYBER-CLEAN GLASSMORPHISM)
        # ---------------------------------------------------------------------
        elif estado == 1:
            header_h = 75
            btn_w = w // (len(CORES) + 2)

            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img, draw=False)

            if len(lmList) == 21:
                x1, y1 = lmList[8][1:]   # Indicador
                x2, y2 = lmList[12][1:]  # Médio
                fingers = detector.fingersUp()

                # Modo Seleção de Cores/Botões (Indicador + Médio erguidos)
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    cv2.rectangle(img, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), drawColor, 2, cv2.LINE_AA)
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 6, (0, 255, 140), -1, cv2.LINE_AA)

                    if y1 < header_h:
                        col_idx = x1 // btn_w
                        if col_idx < len(CORES):
                            cor_atual_idx = col_idx
                            drawColor = CORES[cor_atual_idx]["bgr"]
                        elif col_idx == len(CORES):
                            # Botão Limpar
                            imgCanvas = np.zeros((h, w, 3), np.uint8)
                            msg_status = "Canvas Limpo com Sucesso!"
                            msg_timer = time.time() + 2.0
                        elif col_idx >= len(CORES) + 1:
                            # Botão Finalizar e Gerar QR Code
                            estado = 2
                            timer_sucesso = time.time() + 15.0  # 15s para escanear com calma

                # Modo Pintura (Apenas Indicador erguido)
                elif fingers[1] and not fingers[2]:
                    # Retícula de mira precisa
                    cv2.circle(img, (x1, y1), 8, drawColor, cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(img, (x1, y1), 14, (255, 255, 255), 1, cv2.LINE_AA)

                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if drawColor == (0, 0, 0):
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, eraseThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraseThickness)
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0
            else:
                xp, yp = 0, 0

            # Mescla Canvas na visualização da Câmera
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

            # Botão Finalizar e Gerar QR Code (ENTER)
            bx_send = (len(CORES) + 1) * btn_w
            desenhar_retangulo_arredondado(
                img, (bx_send + 4, 8), (w - 6, header_h - 8),
                cor_fundo=(10, 45, 25), cor_borda=(0, 255, 140), raio=10, alpha=0.92, espessura_borda=2
            )
            cv2.putText(
                img, "QR CODE (ENTER)", (bx_send + 10, header_h // 2 + 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
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
                img, "[1 Dedo: Pintar] | [2 Dedos: Paleta] | [C: Limpar] | [ENTER: QR Code]",
                (770, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.40, (180, 185, 200), 1, cv2.LINE_AA
            )

        # ---------------------------------------------------------------------
        # ESTADO 2: FINALIZAÇÃO, MOLDURA E QR CODE INSTANTÂNEO
        # ---------------------------------------------------------------------
        elif estado == 2:
            if caminho_salvo_recente == "":
                nome_aluno_final = input_nome.strip() or "Visitante ADS"
                timestamp_arq = time.strftime("%Y%m%d_%H%M%S")
                nome_sanitizado = "".join(c for c in nome_aluno_final if c.isalnum() or c in " _-")[:20]
                nome_arq = f"arte_{nome_sanitizado}_{timestamp_arq}.png"
                caminho_salvo_recente = os.path.join(FOLDER_SAVED, nome_arq)

                # Gera a moldura oficial e salva o arquivo
                arte_final = gerar_moldura_oficial(imgCanvas, nome_aluno_final)
                cv2.imwrite(caminho_salvo_recente, arte_final)

                # Registra Lead no CSV de presença
                registrar_lead_csv(nome_aluno_final, nome_arq)

                # 1. URL Local imediata
                ip_local = obter_ip_local()
                url_local = f"http://{ip_local}:{PORTA_LOCAL}/{nome_arq}"
                url_download_atual = url_local
                status_qr_txt = f"Rede do Stand ({ip_local})"
                qr_img_atual = gerar_imagem_qr_code(url_local, tamanho=220)

                # 2. Upload para nuvem em background (para celular em 4G/5G baixar direto)
                upload_nuvem_async(caminho_salvo_recente, atualizar_url_online)

            # Scrim escuro
            scrim = np.full(img.shape, (10, 12, 18), dtype=np.uint8)
            cv2.addWeighted(scrim, 0.85, img, 0.15, 0, img)

            # Card Central Glassmorphism para QR Code
            card_w = 780
            card_h = 380
            cx = w // 2 - card_w // 2
            cy = h // 2 - card_h // 2

            desenhar_retangulo_arredondado(
                img, (cx, cy), (cx + card_w, cy + card_h),
                cor_fundo=(12, 16, 26), cor_borda=(0, 255, 140), raio=16, alpha=0.94, espessura_borda=2
            )

            # LADO ESQUERDO: Textos e Informações
            cv2.putText(
                img, "ARTE PRONTA COM SUCESSO!", (cx + 35, cy + 48),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 140), 2, cv2.LINE_AA
            )
            cv2.line(img, (cx + 35, cy + 68), (cx + card_w - 35, cy + 68), (45, 55, 75), 1)

            cv2.putText(
                img, f"Artista: {input_nome or 'Visitante ADS'}", (cx + 35, cy + 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "Aponte a camera do seu celular:", (cx + 35, cy + 155),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 220, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "Escaneie o QR Code ao lado para baixar", (cx + 35, cy + 190),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, (200, 205, 220), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "sua arte com a moldura oficial de ADS!", (cx + 35, cy + 218),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, (200, 205, 220), 1, cv2.LINE_AA
            )

            # Tag de Status do Link
            cv2.putText(
                img, f"Status: {status_qr_txt}", (cx + 35, cy + 265),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 140), 1, cv2.LINE_AA
            )

            cv2.putText(
                img, "[ESPACO] ou [ENTER] para o proximo aluno", (cx + 35, cy + 305),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 200, 0), 1, cv2.LINE_AA
            )

            # LADO DIREITO: QR Code Renderizado
            if qr_img_atual is not None:
                qrh, qrw = qr_img_atual.shape[:2]
                qrx = cx + card_w - qrw - 35
                qry = cy + 85
                img[qry:qry + qrh, qrx:qrx + qrw] = qr_img_atual
                cv2.rectangle(img, (qrx - 2, qry - 2), (qrx + qrw + 2, qry + qrh + 2), (0, 255, 140), 1, cv2.LINE_AA)
                cv2.putText(
                    img, "ESCANEIE COM O CELULAR", (qrx + 15, qry + qrh + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 255, 140), 1, cv2.LINE_AA
                )

            # Barra de progresso para retorno automático
            rem_t = max(0.0, timer_sucesso - time.time())
            dur_total = 15.0
            prog_t = 1.0 - (rem_t / dur_total)
            bar_y = cy + card_h - 22
            cv2.rectangle(img, (cx + 35, bar_y), (cx + card_w - 35, bar_y + 10), (25, 30, 42), -1)
            cv2.rectangle(img, (cx + 35, bar_y), (cx + 35 + int((card_w - 70) * prog_t), bar_y + 10), (0, 255, 140), -1)

            # Retorno automático após 15 segundos para o próximo aluno
            if time.time() > timer_sucesso:
                estado = 0
                input_nome = ""
                caminho_salvo_recente = ""
                qr_img_atual = None
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
                caminho_salvo_recente = ""
                qr_img_atual = None
            else:
                break
        elif key == 9 or key == ord('f') or key == ord('F'):  # TAB / F
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        # Estado 0: Digitação do Nome
        if estado == 0:
            if key in [13, 10]:  # ENTER
                estado = 1
                imgCanvas = np.zeros((h, w, 3), np.uint8)
            elif key == 32:  # ESPAÇO
                if not input_nome:
                    estado = 1
                    imgCanvas = np.zeros((h, w, 3), np.uint8)
                else:
                    input_nome += " "
            elif key == 8:  # Backspace
                input_nome = input_nome[:-1]
            elif 32 < key <= 126:
                if len(input_nome) < 26:
                    input_nome += chr(key)

        # Estado 1: Pintura
        elif estado == 1:
            if key == 32:  # ESPAÇO: Alternar espelhamento da câmera
                espelhar_video = not espelhar_video
            elif key == ord('c') or key == ord('C'):  # C: Limpar
                imgCanvas = np.zeros((h, w, 3), np.uint8)
                msg_status = "Canvas Limpo!"
                msg_timer = time.time() + 2.0
            elif key in [13, 10]:  # ENTER: Gerar QR Code e salvar
                estado = 2
                timer_sucesso = time.time() + 15.0

        # Estado 2: QR Code na tela
        elif estado == 2:
            if key in [13, 10, 32]:  # ENTER ou ESPAÇO: Avançar para o próximo aluno imediatamente
                estado = 0
                input_nome = ""
                caminho_salvo_recente = ""
                qr_img_atual = None
                imgCanvas = np.zeros((h, w, 3), np.uint8)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
