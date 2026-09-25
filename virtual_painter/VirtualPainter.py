"""
Ponto de entrada do Virtual Painter.
Executa a versão moderna com cadastro de alunos, moldura de ADS e envio por e-mail.
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from VirtualPainter_Arcade import main

if __name__ == "__main__":
    main()
