"""
=============================================================================
     STAND ADS - UNIMAR ABERTA (PAINEL PRINCIPAL / LAUNCHER)
=============================================================================
Menu unificado para demonstração interativa dos projetos de Visão Computacional
e Inteligência Artificial no evento Unimar Aberta:
  [1] Snake Donuts - Arcade Edition (Viciante, competitivo, neon & ranking)
  [2] Virtual Painter - Neon Edition (Pintura no ar com gestos e salvar arte)
  [3] IA Reconhecedor LIBRAS (Classificador com Deep Learning & Desafio)
  [Q] Sair
=============================================================================
"""

import sys
import os
import subprocess
import time
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJETOS = [
    {
        "id": "1",
        "titulo": "1. SNAKE DONUTS ARCADE",
        "tag": "JOGO COMPETITIVO & VICIANTE",
        "desc": "Cobra neon, combos x1-x5, particulas, power-ups e Ranking Top 5 com nomes.",
        "script": os.path.join(SCRIPT_DIR, "Snake_Donut", "Snake_Arcade_Edition.py"),
        "cor": (0, 255, 120)
    },
    {
        "id": "2",
        "titulo": "2. VIRTUAL PAINTER NEON",
        "tag": "CRIATIVIDADE & INTERATIVIDADE",
        "desc": "Desenho no ar com as maos, moldura oficial de ADS e QR Code instantaneo no celular.",
        "script": os.path.join(SCRIPT_DIR, "virtual_painter", "VirtualPainter_Arcade.py"),
        "cor": (255, 180, 0)
    },
    {
        "id": "3",
        "titulo": "3. IA EDUCACIONAL LIBRAS",
        "tag": "IA EDUCACIONAL & ACESSIBILIDADE",
        "desc": "Alfabeto completo (A-Z), 3 Modos (Tela Limpa, Desafio, Soletrar no Ar) e Easter Eggs.",
        "script": os.path.join(SCRIPT_DIR, "libras", "libras_stand_edition.py"),
        "cor": (0, 215, 255)
    },
    {
        "id": "4",
        "titulo": "4. MATH BLITZ - DESAFIO DOS DEDOS",
        "tag": "AGILIDADE MENTAL & REFLEXOS",
        "desc": "Resolva continhas rapidas no ar mostrando a quantidade de dedos em 45 segundos!",
        "script": os.path.join(SCRIPT_DIR, "finger_arcade", "Finger_Arcade.py"),
        "cor": (255, 120, 220)
    }
]

def render_menu():
    largura = 1200
    altura = 740
    tela = np.zeros((altura, largura, 3), dtype=np.uint8)

    # Gradiente de fundo sutil
    for y in range(altura):
        val = int(15 + (y / altura) * 20)
        tela[y, :] = (val, val, val + 10)

    # Linhas e acentos de design
    cv2.line(tela, (60, 115), (largura - 60, 115), (0, 220, 255), 2)
    cv2.line(tela, (60, altura - 70), (largura - 60, altura - 70), (80, 80, 100), 1)

    # Topo / Header
    cv2.putText(
        tela, "ANALISE E DESENVOLVIMENTO DE SISTEMAS", (65, 55),
        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        tela, "UNIMAR ABERTA * SHOWCASE DE VISAO COMPUTACIONAL", (65, 95),
        cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA
    )

    # Cards dos Projetos
    card_w = largura - 120
    card_h = 108
    start_y = 135
    espacamento = 18

    for i, proj in enumerate(PROJETOS):
        y_pos = start_y + i * (card_h + espacamento)

        # Fundo do Card
        cv2.rectangle(tela, (60, y_pos), (60 + card_w, y_pos + card_h), (30, 32, 42), -1)
        cv2.rectangle(tela, (60, y_pos), (60 + card_w, y_pos + card_h), proj["cor"], 2)

        # Barra lateral colorida
        cv2.rectangle(tela, (60, y_pos), (72, y_pos + card_h), proj["cor"], -1)

        # Título
        cv2.putText(
            tela, proj["titulo"], (95, y_pos + 38),
            cv2.FONT_HERSHEY_DUPLEX, 0.88, proj["cor"], 2, cv2.LINE_AA
        )

        # Tag
        cv2.putText(
            tela, f"[ {proj['tag']} ]", (95 + 460, y_pos + 36),
            cv2.FONT_HERSHEY_DUPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Descrição
        cv2.putText(
            tela, proj["desc"], (95, y_pos + 76),
            cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
        )

        # Botão de Ação
        btn_txt = f"PRESSIONE [{proj['id']}]"
        cv2.putText(
            tela, btn_txt, (60 + card_w - 240, y_pos + 78),
            cv2.FONT_HERSHEY_DUPLEX, 0.68, proj["cor"], 2, cv2.LINE_AA
        )

    # Rodapé / Instruções
    cv2.putText(
        tela, "Pressione [1, 2, 3 ou 4] no teclado para iniciar  |  [F: Tela Cheia]  |  [Q ou ESC: Sair]",
        (largura // 2 - 450, altura - 30),
        cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA
    )

    return tela

def executar_projeto(caminho_script):
    if not os.path.exists(caminho_script):
        print(f"[ERRO] Script nao encontrado: {caminho_script}")
        return
    print(f"\n[INICIANDO] {caminho_script}...")
    cv2.destroyAllWindows()
    # Executa o processo e aguarda finalizar para voltar ao menu
    subprocess.run([sys.executable, caminho_script])
    print("[RETORNANDO] Ao menu principal do stand...")

def main():
    nome_janela = "STAND ADS - UNIMAR ABERTA | Hub de Demonstracoes"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)

    # Inicia em tela cheia para o stand
    fullscreen = True
    cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    img_menu = render_menu()

    while True:
        cv2.imshow(nome_janela, img_menu)
        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == 9 or key == ord('f') or key == ord('F'):  # TAB / F: Tela Cheia
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == ord('1'):
            executar_projeto(PROJETOS[0]["script"])
            cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord('2'):
            executar_projeto(PROJETOS[1]["script"])
            cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord('3'):
            executar_projeto(PROJETOS[2]["script"])
            cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord('4'):
            executar_projeto(PROJETOS[3]["script"])
            cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
