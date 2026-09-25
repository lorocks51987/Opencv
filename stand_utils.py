"""
=============================================================================
     STAND UTILS - MÓDULO COMPARTILHADO (ADS UNIMAR ABERTA)
=============================================================================
Utilitários visuais e de infraestrutura compartilhados entre todos os projetos
do stand para eliminar duplicação de código e garantir consistência visual.
=============================================================================
"""

import cv2
import numpy as np
import os
import threading

# =============================================================================
# GLASSMORPHISM CARD (VERSÃO ÚNICA, OTIMIZADA)
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


# =============================================================================
# BUSCA INTELIGENTE DE CÂMERA (UNIFICADA)
# =============================================================================
def encontrar_camera(indices=[1, 0, 2]):
    """Busca uma webcam funcional nos índices fornecidos."""
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[OK] Câmera conectada no índice: {idx}")
                return cap
            cap.release()
    return None


# =============================================================================
# SONS PROCEDURAIS SEGUROS EM BACKGROUND (THREAD)
# =============================================================================
try:
    import winsound

    def _beep_bg(*beeps):
        """Toca uma sequência de beeps em thread de background."""
        def _worker():
            for freq, dur in beeps:
                winsound.Beep(freq, dur)
        threading.Thread(target=_worker, daemon=True).start()

    def som_acerto():
        _beep_bg((988, 70), (1318, 110))

    def som_erro():
        _beep_bg((440, 90), (370, 90))

    def som_tick():
        _beep_bg((880, 40),)

    def som_vitoria():
        _beep_bg((523, 60), (659, 60), (784, 60), (1046, 60))

    def som_fim():
        _beep_bg((440, 90), (370, 90), (311, 90))

except ImportError:
    def som_acerto(): pass
    def som_erro(): pass
    def som_tick(): pass
    def som_vitoria(): pass
    def som_fim(): pass


# =============================================================================
# RODAPÉ PADRONIZADO CYBER-CLEAN (38px)
# =============================================================================
def desenhar_rodape(img, titulo_projeto, espelhar_video, extras=None):
    """Desenha o rodapé padronizado de 38px no fundo da imagem.
    
    Args:
        img: Imagem BGR
        titulo_projeto: Nome do projeto (ex: "MATH BLITZ")
        espelhar_video: Estado do espelhamento
        extras: Lista opcional de tuplas (texto, cor) para informações adicionais
    """
    h, w = img.shape[:2]
    foot_h = 38
    foot_roi = img[h - foot_h:h, 0:w]
    foot_bg = np.full(foot_roi.shape, (10, 12, 18), dtype=np.uint8)
    cv2.addWeighted(foot_bg, 0.88, foot_roi, 0.12, 0, foot_roi)

    cv2.putText(
        img, f"ADS * UNIMAR ABERTA | {titulo_projeto}", (20, h - 14),
        cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 140), 1, cv2.LINE_AA
    )

    status_espelho = "LIGADO" if espelhar_video else "DESLIGADO"
    cor_espelho = (0, 255, 140) if espelhar_video else (0, 220, 255)
    cv2.putText(
        img, f"[ESPACO] Espelho: {status_espelho}", (365, h - 14),
        cv2.FONT_HERSHEY_DUPLEX, 0.44, cor_espelho, 1, cv2.LINE_AA
    )
