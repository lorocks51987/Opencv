# 🎪 Showcase de Visão Computacional e IA - ADS Unimar Aberta

Projetos interativos de Visão Computacional e Inteligência Artificial desenvolvidos para o stand do curso de **Análise e Desenvolvimento de Sistemas (ADS)** no evento **Unimar Aberta**.

Todos os projetos foram padronizados com a mesma identidade visual **Cyber-Clean Glassmorphism**, rodapé de status, suporte a **Iriun Webcam**, atalhos universais e performance otimizada a **60 FPS**.

---

## 🕹️ Launcher do Stand (Painel Principal)

Para facilitar a apresentação durante o evento sem necessidade de comandos no terminal para cada visitante, utilize o launcher unificado:

```bash
python menu_stand_unimar.py
```

No menu principal, basta pressionar a tecla correspondente:
- **`[1]` 🐍 Snake Donuts - Arcade Edition**
- **`[2]` 🎨 Virtual Painter - Neon Edition**
- **`[3]` 🔤 IA Educacional de LIBRAS**
- **`[Q / ESC]`** Fechar Menu

---

## 🧭 Padrão Universal de Atalhos do Stand

Todos os módulos seguem convenções unificadas de teclado e display para facilitar a operação pelos alunos e monitores no stand:

| Tecla / Atalho | Função Universal | Descrição |
| :--- | :--- | :--- |
| **`[TAB]`** ou **`[F]`** | **Tela Cheia (Fullscreen)** | Alterna entre janela e tela cheia para TV ou monitor do stand |
| **`[ESPAÇO]`** | **Inverter Espelho da Câmera** | Alterna entre câmera normal e espelhada (ideal para Iriun Webcam) |
| **`[ESC]`** ou **`[Q]`** | **Sair / Voltar ao Menu** | Fecha o módulo atual com segurança e libera a câmera |
| **Rodapé Inferior** | **Barra Cyber-Clean** | Exibe atalhos ativos, status do espelho, áudio e FPS em tempo real |

---

## 🌟 Detalhes dos Projetos em Destaque

### 1. 🐍 Snake Donuts (Arcade Edition)
*Localização: `Snake_Donut/Snake_Arcade_Edition.py` ou `Snake_Donut/main.py`*

Uma experiência arcade retro-futurista altamente competitiva e viciante para atrair e engajar os visitantes:
- **Mecânica de Combos (`x1` até `x5`)**: Comer itens em rápida sequência ativa multiplicadores com barra de tempo de adrenalina.
- **Game Juice & Partículas**: Explosões de partículas coloridas temáticas ao coletar donuts, maçãs e moedas.
- **Rastro Neon & Glow**: Cobra com gradiente luminoso, cabeça com efeito de luz e olhos expressivos direcionados à comida.
- **Itens e Power-ups**:
  - 🍩 **Donut (+10 pts)**: Cresce a cobra e pontua.
  - 🍎 **Maçã Encantada (+25 pts)**: Concede 6s de **Invencibilidade** (a cobra brilha em arco-íris e atravessa o próprio corpo).
  - 🧪 **Poção (+15 pts)**: Encurta a cobra pela metade para facilitar manobras apertadas.
  - 🪙 **Moeda (+50 pts)**: Dá pontos extras e liberta o Fantasma bônus.
  - 👻 **Fantasma (+100 pts)**: Atravessa a tela em velocidade; colete-o para pontuação máxima.
- **Placar Minimalista & Direto**: Exibe de forma limpa e impactante o **Score do Aluno** em verde neon e o **Maior Recorde do Stand** em dourado (com celebração e salvamento automático caso o recorde seja superado).
- **Controles Gestuais & Reinício Rápido**: Movimento guiado pela ponta do indicador (com suavização anti-ruído) e reinício instantâneo fechando a mão 2 vezes ou teclando `[R]` / `[ESPAÇO]`.

#### ⌨️ Tabela Completa de Atalhos - Snake Donuts:
| Tecla | Ação |
| :--- | :--- |
| **`[ESPAÇO]`** | Alterna espelhamento da câmera durante o jogo / Reinicia a partida em Game Over |
| **`[R]`** | Reinicia a partida imediatamente a qualquer momento |
| **`[TAB]`** ou **`[F]`** | Alterna Modo Tela Cheia |
| **`[M]`** | Alterna Som (Mudo / Ativado) |
| **`[ESC]`** ou **`[Q]`** | Sair do jogo |

---

### 2. 🎨 Virtual Painter (Edição Stand com QR Code Instantâneo)
*Localização: `virtual_painter/VirtualPainter_Arcade.py` ou `virtual_painter/VirtualPainter.py`*

Permite aos visitantes desenharem no ar como mágica e levarem sua obra para casa na hora apontando a câmera do celular:
- **Entrada Express**: Cadastro ultrarrápido apenas com Nome/Apelido (ou pressione Espaço para entrar como Visitante direto).
- **Espelhamento de Câmera Natural**: O visitante se movimenta de forma intuitiva como diante de um espelho interativo.
- **Modos Gestuais**:
  - ☝️ **1 Dedo (Indicador)**: Desenha na tela com traço suave e anti-tremor.
  - ✌️ **2 Dedos (Indicador + Médio)**: Modo Seleção: permite escolher cores no topo ou clicar nos botões.
- **Paleta de Cores**: Verde Neon, Rosa Choque, Azul Ciano, Amarelo Ouro, Branco e Borracha.
- **Moldura Oficial de ADS Unimar**: A arte é enquadrada automaticamente em alta definição com logo, data e nome do aluno.
- **QR Code Instantâneo na Tela Final**: O visitante aponta o celular e baixa a foto na hora:
  - 🌐 **Nuvem (Download Direto)**: Funciona em qualquer rede 4G/5G ou Wi-Fi.
  - 🏠 **Rede Local Stand**: Micro-servidor HTTP embutido para download direto sem precisar de internet externa.
- **Lista de Presença (`leads_visitantes.csv`)**: Registra a lista de alunos que pintaram durante o evento.

#### ⌨️ Tabela Completa de Atalhos - Virtual Painter:
| Tecla | Ação |
| :--- | :--- |
| **`[ENTER]`** | Confirma o Nome (no cadastro) / Finaliza e Gera o QR Code (na pintura) |
| **`[ESPAÇO]`** | Inicia como Visitante direto (no cadastro) / Inverte espelho (na pintura) / Próximo aluno (no QR Code) |
| **`[F]`** ou **`[TAB]`** | Alterna Modo Tela Cheia a qualquer momento |
| **`[C]`** | Limpa todo o desenho da tela |
| **`[ESC]`** | Volta à tela de boas-vindas ou encerra o aplicativo |

---

### 3. 🔤 IA Educacional de LIBRAS (3 Modos Interativos & Easter Eggs)
*Localização: `libras/libras_stand_edition.py` ou `libras/libras_classifier.py`*

Demonstração do potencial acadêmico, social e de inclusão da Inteligência Artificial:
- **3 Modos de Apresentação Dedicados**:
  - 🌿 **Modo 0 (`[F1]`): Tela Limpa (Minimalista / Zen)**: Apenas a câmera widescreen com uma pílula moderna translúcida no canto superior direito mostrando a letra reconhecida com glow. Zero poluição visual!
  - 🎯 **Modo 1 (`[F2]`): Jogo Desafio Educativo**: Desafio de acertar a letra sorteada, com guia oficial de como posicionar cada dedo, barra de 1s e placar de acertos do aluno!
  - ✍️ **Modo 2 (`[F3]`): Soletrador de Palavras no Ar**: O visitante soletra palavras ou o próprio nome no ar em LIBRAS! Ao segurar uma letra por 1s, ela é adicionada à palavra na tela (ex: `L-U-C-A-S`).
- **Exclusividade da Mão Direita (Anti-Confusão)**: Reconhece e classifica apenas a mão direita física, eliminando qualquer inversão ou confusão entre letras espelhadas (ex: `O`, `C`, `D`). Caso o visitante use a mão esquerda, a IA orienta visualmente a levantar a mão direita.
- **Classificador de Machine Learning Vectorized (KNN)**: 73 características biométricas 3D com processamento BLAS/SIMD em C (latência de **0.14 ms** e taxa de **60 FPS** estável).
- **Alfabeto Completo (A a Z) com Salvamento Permanente no Disco**: Calibre qualquer letra em menos de 1 segundo pressionando a respectiva tecla (`A` a `Z`). O modelo atualiza e grava imediatamente no arquivo `libras_dataset.json` de forma persistente!
- **Easter Eggs Especiais**:
  - 🔞 **Dedo do Meio (🖕)**: Efeito censura de TV (mosaico pixelado na mão, tarja vermelha `[ CENSURADO ]` e som clássico de PIIII).
  - 👍 **Joinha**: Chuva de confetes coloridos, selo `100% APROVADO PELO STAND DE ADS!` e jingle triunfal (com filtro biométrico inteligente para **nunca** confundir com a letra "A").

#### ⌨️ Tabela Completa de Atalhos - IA de LIBRAS:
| Tecla | Ação |
| :--- | :--- |
| **`[F1]`** | Ativa o **Modo 0: Tela Limpa** (Zen / Minimalista) |
| **`[F2]`** | Ativa o **Modo 1: Jogo Desafio** (Stand Educativo com Placar) |
| **`[F3]`** | Ativa o **Modo 2: Soletrador de Palavras** (Escreva seu Nome no Ar) |
| **`[ESPAÇO]`** | Inverte o espelho da câmera (essencial para Iriun Webcam) |
| **`[TAB]`** | Alterna Modo Tela Cheia |
| **`[A a Z]`** | Ensina / Calibra a respectiva letra para o usuário em 2 segundos |
| **`[1]`** | Calibra o gesto do Joinha |
| **`[2]`** | Calibra o gesto da Censura |
| **`[BACKSPACE]`** | Apaga a última letra digitada no Modo Soletrador |
| **`[DELETE]`** | Limpa toda a palavra no Modo Soletrador |
| **`[ESC]`** | Sair da IA |

---

### 4. 🧮 Math Blitz: Desafio dos Dedos (Edição Stand)
*Localização: `finger_arcade/Finger_Arcade.py` ou `examples/countFinger.py`*

Jogo arcade eletrizante focado em agilidade mental, matemática rápida e reflexos biométricos:
- **Resolução de Contas no Ar**: O aluno resolve continhas matemáticas em tempo real mostrando a quantidade exata de dedos (0 a 10 dedos usando ambas as mãos simultaneamente).
- **Desafios Dinâmicos**:
  - Somas rápidas: *"QUANTO É: 4 + 3?"*, *"5 + 2 = ?"*, *"1 + 4 = ?"*
  - Subtrações rápidas: *"QUANTO É: 8 - 3?"*, *"10 - 4 = ?"*
  - Reação direta: *"MOSTRE EXATAMENTE: 7 DEDOS!"*, *"MOSTRE: 10 DEDOS!"*
- **Combos e Efeitos Visuais**: Acertos rápidos sustentados por 0.3s disparam bônus de multiplicador de Combo (**x1 a x5**), explosão de partículas neon e sons procedurais.
- **Partidas de 45 Segundos**: Tempo ideal para dinâmica rápida de filas no stand, permitindo que vários visitantes compitam em sequência.
- **Placar Minimalista do Stand**: Tela de Game Over com a pontuação final do aluno e o **Maior Recorde de Todo o Evento** salvo automaticamente no disco (`recorde_math.json`).

#### ⌨️ Tabela Completa de Atalhos - Math Blitz:
| Tecla | Ação |
| :--- | :--- |
| **`[ESPAÇO]`** | Inicia a partida / Reinicia em Game Over / Alterna espelhamento da câmera |
| **`[R]`** | Reinicia imediatamente a partida a qualquer momento |
| **`[TAB]`** ou **`[F]`** | Alterna Modo Tela Cheia |
| **`[ESC]`** ou **`[Q]`** | Sair do jogo |

---

## 🛠️ Instalação e Execução

### 1. Inicializadores de 1 Clique (Recomendado para o Stand)
Criamos executáveis `.bat` prontos para rodar em qualquer máquina Windows:
- 🚀 **`INSTALAR_TUDO.bat`**: Instala e atualiza automaticamente todas as dependências com 1 clique.
- 🎮 **`INICIAR_STAND.bat`**: Abre o Menu Launcher geral com acesso aos 4 projetos do stand.
- 🐍 **`INICIAR_COBRINHA.bat`**: Inicia diretamente o Snake Donuts Arcade.
- 🔤 **`INICIAR_LIBRAS.bat`**: Inicia diretamente a IA de LIBRAS Stand Edition.
- 🎨 **`INICIAR_PINTOR.bat`**: Inicia diretamente o Pintor Virtual com QR Code.
- 🖐️ **`INICIAR_CONTADOR.bat`**: Inicia diretamente o Finger Arcade & Gesture Arena.

### 2. Execução Manual via Terminal
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Iniciar Launcher do Stand
python menu_stand_unimar.py
```

---

## 💡 Dicas de Sucesso para o Stand na Unimar Aberta

1. **Iluminação Frontal**: Evite luzes fortes ou janelas diretamente atrás do visitante (contra-luz). Uma luz frontal ou iluminação padrão de sala permite que o MediaPipe rastreie todos os 21 pontos da mão perfeitamente.
2. **Distância Ideal**: Posicione a câmera na altura do peito, a cerca de **1,0 a 1,5 metro** do visitante.
3. **Iriun Webcam**: Se utilizar o celular como webcam via Iriun, caso a imagem pareça invertida, basta pressionar **`[ESPAÇO]`** uma vez em qualquer um dos programas para corrigir instantaneamente.
4. **Competitividade com o Recorde**: No jogo da Cobrinha, a tela de Game Over destaca a pontuação do aluno versus o **Maior Recorde de Todo o Evento**, motivando filas de estudantes querendo superar a marca máxima do stand!
5. **Engajamento com o QR Code**: No Pintor Virtual, motive o visitante a apontar a câmera do celular para o QR Code da tela final para baixar a foto na hora e postar no Instagram marcando `@unimar`!
