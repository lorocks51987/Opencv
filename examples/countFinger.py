"""
Ponto de entrada alternativo para o Math Blitz: Desafio dos Dedos.
Executa o desafio de 45 segundos resolvendo continhas no ar (0 a 10 dedos).
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
