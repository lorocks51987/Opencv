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

### 2. 🎨 Virtual Painter (Edição Stand com Envio por E-mail)
*Localização: `virtual_painter/VirtualPainter_Arcade.py` ou `virtual_painter/VirtualPainter.py`*

Permite aos visitantes desenharem no ar como mágica e levarem sua obra para casa:
- **Tela de Boas-Vindas & Cadastro**: Coleta o Nome e E-mail do aluno antes de iniciar a pintura (pressione Enter para avançar rapidamente).
- **Espelhamento de Câmera Natural**: O visitante se movimenta de forma intuitiva como diante de um espelho interativo.
- **Modos Gestuais**:
  - ☝️ **1 Dedo (Indicador)**: Desenha na tela com traço suave e anti-tremor.
  - ✌️ **2 Dedos (Indicador + Médio)**: Modo Seleção: permite escolher cores no topo ou clicar nos botões.
- **Paleta de Cores**: Verde Neon, Rosa Choque, Azul Ciano, Amarelo Ouro, Branco e Borracha.
- **Moldura Oficial de ADS Unimar**: A arte é enquadrada automaticamente com logo e dados do aluno.
- **Envio Automático por E-mail**: Disparo assíncrono (sem travar a câmera) com anexo da imagem para o visitante.
- **Geração de Leads (`leads_visitantes.csv`)**: Salva a lista de contatos de todos os visitantes para o curso.
- **Fila Offline (`fila_emails.json`)**: Se o stand ficar sem internet no momento, todas as artes ficam armazenadas em fila para envio posterior.

#### ⌨️ Tabela Completa de Atalhos - Virtual Painter:
| Tecla | Ação |
| :--- | :--- |
| **`[ENTER]`** | Confirma Nome/E-mail no cadastro ou Finaliza e Envia a pintura |
| **`[TAB]`** | Alterna entre campo de Nome e E-mail (no cadastro) / Alterna Tela Cheia (na pintura) |
| **`[F]`** | Alterna Modo Tela Cheia a qualquer momento |
| **`[ESPAÇO]`** | Inicia como Visitante direto (no cadastro) / Inverte o espelhamento da câmera (na pintura) |
| **`[C]`** | Limpa todo o desenho da tela |
| **`[ESC]`** | Volta à tela de boas-vindas (se estiver pintando) ou fecha o aplicativo |

---

### 3. 🔤 IA Educacional de LIBRAS (3 Modos Interativos & Easter Eggs)
*Localização: `libras/libras_stand_edition.py` ou `libras/libras_classifier.py`*

Demonstração do potencial acadêmico, social e de inclusão da Inteligência Artificial:
- **3 Modos de Apresentação Dedicados**:
  - 🌿 **Modo 0 (`[F1]`): Tela Limpa (Minimalista / Zen)**: Apenas a câmera widescreen com uma pílula moderna translúcida no canto superior direito mostrando a letra reconhecida com glow. Zero poluição visual!
  - 🎯 **Modo 1 (`[F2]`): Jogo Desafio Educativo**: Desafio de acertar a letra sorteada, com guia oficial de como posicionar cada dedo, barra de 1s e placar de acertos do aluno!
  - ✍️ **Modo 2 (`[F3]`): Soletrador de Palavras no Ar**: O visitante soletra palavras ou o próprio nome no ar em LIBRAS! Ao segurar uma letra por 1s, ela é adicionada à palavra na tela (ex: `L-U-C-A-S`).
- **Classificador de Machine Learning Vectorized (KNN)**: 73 características biométricas 3D com processamento BLAS/SIMD em C (latência de **0.14 ms** e taxa de **60 FPS** estável).
- **Alfabeto Completo (A a Z)** com modo de calibração em 2 segundos (teclas `A` a `Z` 100% livres, sem qualquer colisão de comandos!).
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

## 🛠️ Instalação e Execução

### 1. Pré-requisitos
- Python 3.8+ (Recomendado: Python 3.10)
- Webcam (resolução 720p ou 1080p, ou smartphone conectado via Iriun Webcam)

### 2. Instalação das dependências
```bash
pip install -r requirements.txt
```

---

## 💡 Dicas de Sucesso para o Stand na Unimar Aberta

1. **Iluminação Frontal**: Evite luzes fortes ou janelas diretamente atrás do visitante (contra-luz). Uma luz frontal ou iluminação padrão de sala permite que o MediaPipe rastreie todos os 21 pontos da mão perfeitamente.
2. **Distância Ideal**: Posicione a câmera na altura do peito, a cerca de **1,0 a 1,5 metro** do visitante.
3. **Iriun Webcam**: Se utilizar o celular como webcam via Iriun, caso a imagem pareça invertida, basta pressionar **`[ESPAÇO]`** uma vez em qualquer um dos programas para corrigir instantaneamente.
4. **Competitividade com o Recorde**: No jogo da Cobrinha, a tela de Game Over destaca a pontuação do aluno versus o **Maior Recorde de Todo o Evento**, motivando filas de estudantes querendo superar a marca máxima do stand!
5. **Captura de Contatos**: No Pintor Virtual, garanta que os visitantes preencham o e-mail para receberem a arte com a moldura oficial do curso de ADS da Unimar.
