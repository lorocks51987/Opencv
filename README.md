# 🎮 Stand de Visão Computacional — ADS UNIMAR ABERTA

Conjunto de **4 projetos interativos** de Visão Computacional e Inteligência Artificial
desenvolvidos para os stands do curso de **Análise e Desenvolvimento de Sistemas** na
**Unimar Aberta**.

Todos os projetos utilizam **OpenCV**, **MediaPipe** e **webcam** para criar experiências
onde os visitantes controlam tudo com as mãos — sem teclado, sem mouse.

---

## 📋 Projetos do Stand

| # | Projeto | Descrição | Controle |
|---|---------|-----------|----------|
| 1 | **Snake Donuts Arcade** | Cobra neon com combos x1-x5, power-ups, fantasmas e ranking Top 5 | Dedo indicador |
| 2 | **Virtual Painter Neon** | Pintura no ar com QR Code instantâneo para levar a arte no celular | 1 dedo = pinta, 2 dedos = paleta |
| 3 | **IA Educacional LIBRAS** | Reconhecedor de sinais de LIBRAS com 3 modos (Desafio, Soletrador, Zen) | Mão direita |
| 4 | **Math Blitz** | Resolva continhas rápidas mostrando a resposta com os dedos (45s) | Duas mãos (0-10 dedos) |

---

## 🚀 Início Rápido

### 1. Instalar Dependências
```
INSTALAR_TUDO.bat    (duplo clique — cria .venv e instala tudo)
```

### 2. Iniciar o Stand
```
INICIAR_STAND.bat    (abre o hub de seleção de projetos em tela cheia)
```

### 3. Ou iniciar cada projeto individualmente
```
INICIAR_COBRINHA.bat
INICIAR_PINTOR.bat
INICIAR_LIBRAS.bat
INICIAR_CONTADOR.bat
```

---

## 🎛️ Atalhos Universais (Todos os Projetos)

| Tecla | Ação |
|-------|------|
| `ESC` | Sair / Voltar |
| `ESPAÇO` | Ação contextual (iniciar partida, espelhar câmera) |
| `TAB` / `F` | Alternar tela cheia |
| `R` | Reiniciar partida |

### Atalhos Específicos

**LIBRAS:**
- `F1` / `F2` / `F3` — Trocar modo (Zen / Desafio / Soletrador)
- `A-Z` — Calibrar letra ao vivo
- `1` / `2` — Calibrar easter eggs (Joinha / Censura)

**Snake Donuts:**
- `M` — Mute / unmute sons

**Virtual Painter:**
- `C` — Limpar canvas
- `ENTER` — Gerar QR Code e salvar arte

---

## 🐣 Easter Eggs

- 🖕 **Dedo do Meio** → Efeito de censura de TV (pixelate + tarja + som PIIII)
- 👍 **Joinha** → Chuva de confetes neon + selo "100% APROVADO PELO STAND"

> Ambos funcionam no projeto de LIBRAS. São calibráveis (teclas `1` e `2`).

---

## 📁 Estrutura do Projeto

```
Opencv/
├── INSTALAR_TUDO.bat           # Instalador de dependências
├── INICIAR_STAND.bat           # Hub principal do stand
├── INICIAR_COBRINHA.bat        # Launcher Snake Donuts
├── INICIAR_PINTOR.bat          # Launcher Virtual Painter
├── INICIAR_LIBRAS.bat          # Launcher LIBRAS
├── INICIAR_CONTADOR.bat        # Launcher Math Blitz
├── menu_stand_unimar.py        # Hub de seleção de projetos
├── stand_utils.py              # Utilitários visuais compartilhados
├── requirements.txt            # Dependências Python
│
├── Snake_Donut/
│   ├── Snake_Arcade_Edition.py # Código principal do Snake
│   ├── leaderboard.json        # Ranking Top 5 (gerado em runtime)
│   ├── Donut.png / coin.png / ghost*.png / Potion.png / enchanted_apple.gif
│
├── virtual_painter/
│   ├── VirtualPainter_Arcade.py  # Código principal do Pintor
│   ├── HandTrackingModule.py     # Módulo de detecção de mãos
│   ├── galeria_visitantes/       # Artes salvas dos visitantes
│   └── leads_visitantes.csv      # Log de presença (gerado em runtime)
│
├── libras/
│   ├── libras_stand_edition.py   # Código principal do LIBRAS
│   ├── libras_ml_engine.py       # Motor KNN de classificação
│   └── libras_dataset.json       # Dataset de landmarks (calibrável)
│
├── finger_arcade/
│   ├── Finger_Arcade.py          # Código principal do Math Blitz
│   └── recorde_math.json         # Recorde do stand (gerado em runtime)
│
└── examples/
    ├── countFinger.py            # Wrapper alternativo para o Math Blitz
    └── control_mouse.py          # Demo de controle de mouse por gestos
```

---

## 🔧 Requisitos do Sistema

- **Python** 3.10+ (com "Add to PATH" marcado na instalação)
- **Webcam** USB ou integrada (resolução mínima: 640x480)
- **Windows** 10 ou 11 (sons via `winsound`, câmera via `DirectShow`)
- **Pacotes Python**: opencv-python, opencv-contrib-python, mediapipe, cvzone, pillow, numpy

---

## 📜 Licença

Projeto acadêmico para fins educacionais — **Curso de ADS, Universidade de Marília (UNIMAR)**.
