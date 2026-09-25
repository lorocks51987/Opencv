"""
Ponto de entrada do Finger Arcade & Gesture Arena.
Executa a versão definitiva com 3 Modos:
  [F1] Dashboard Biométrico Sci-Fi (0 a 10 Dedos)
  [F2] Math Blitz & Desafio de Reação Rápida (45s)
  [F3] Jokenpô contra o Robô de ADS (Pedra, Papel e Tesoura)
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
FINGER_DIR = os.path.join(ROOT_DIR, "finger_arcade")

if FINGER_DIR not in sys.path:
    sys.path.insert(0, FINGER_DIR)

from Finger_Arcade import main

if __name__ == "__main__":
    main()
