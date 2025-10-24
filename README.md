# 🤚 Hand Gesture Recognition Projects

Coleção de projetos de reconhecimento de gestos com as mãos usando visão computacional e machine learning.

## 📋 Projetos Incluídos

### 🔤 LIBRAS - Reconhecimento de Sinais
- **`libras/libras_classifier.py`** - Classificador de sinais LIBRAS (A, E, I, O, U)
- **`libras/libras_data_collector.py`** - Coleta de dados para treinamento
- **`libras/Model/`** - Modelo treinado (keras_model.h5 + labels.txt)

### 🎨 Virtual Painter - Pintura com Gestos
- **`virtual_painter/VirtualPainter.py`** - Pintura virtual usando gestos da mão
- **`virtual_painter/HandTrackingModule.py`** - Módulo de detecção de mãos

### 🖱️ Examples - Exemplos de Uso
- **`examples/control_mouse.py`** - Controle de mouse com gestos
- **`examples/countFinger.py`** - Contagem de dedos em tempo real

## 🚀 Instalação Rápida

### Pré-requisitos
- Python 3.8 ou superior
- Câmera web ou dispositivo de captura
- Sistema operacional: Windows, macOS ou Linux

### Instalação das Dependências

```bash
# Clone o repositório
git clone <seu-repositorio>
cd hands

# Instalar dependências

```

### ⚠️ Problemas Comuns

**Erro do TensorFlow:**
```bash
pip uninstall tensorflow -y
pip install tensorflow==2.15.0
```

**Erro "Long Path" no Windows:**
```bash
pip install opencv-python==4.8.0.76 numpy==1.24.3 mediapipe==0.10.3 cvzone==1.5.6 tensorflow==2.15.0 pyautogui==0.9.54
```

### Instalação com Ambiente Virtual (Recomendado)

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Instalar dependências

```

## 🎯 Como Usar

### 1. Reconhecimento LIBRAS
```bash
cd libras
python libras_classifier.py
```
- Mostra sinais A, E, I, O, U na câmera
- O sistema detecta e classifica automaticamente

### 2. Pintura Virtual
```bash
cd virtual_painter
python VirtualPainter.py
```
- **Gestos**: 
  - 2 dedos levantados = Seleção de cor
  - 1 dedo levantado = Desenhar
- **Cores**: Verde, Vermelho, Azul, Borracha

### 3. Controle de Mouse
```bash
cd examples
python control_mouse.py
```
- Mova o dedo indicador para mover o cursor
- Toque polegar + indicador para clicar

### 4. Contagem de Dedos
```bash
cd examples
python countFinger.py
```
- Detecta e conta dedos levantados em tempo real

## ⚙️ Configuração da Câmera

### Detecção Automática
A maioria dos projetos detecta automaticamente a câmera disponível.

### Configuração Manual
Se necessário, ajuste o índice da câmera nos arquivos:
```python
cap = cv2.VideoCapture(0)  # Câmera padrão
cap = cv2.VideoCapture(1)  # Câmera externa
```

### Câmeras Externas (Iriun, DroidCam, etc.)
Os projetos são compatíveis com câmeras externas via aplicativos como:
- Iriun Webcam
- DroidCam
- EpocCam

## 🔧 Solução de Problemas

### Erro: "Nenhuma câmera encontrada"
1. Verifique se a câmera não está sendo usada por outro programa
2. Teste diferentes índices de câmera (0, 1, 2...)
3. Reinicie o aplicativo de câmera externa

### Erro: "ModuleNotFoundError"
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Performance Lenta
- Feche outros programas que usam a câmera
- Reduza a resolução da câmera no código
- Use uma câmera com melhor qualidade

## 📦 Dependências

- **opencv-python** - Processamento de imagem
- **numpy** - Operações matemáticas
- **mediapipe** - Detecção de mãos
- **cvzone** - Classificação de gestos
- **tensorflow** - Machine learning
- **pyautogui** - Controle de mouse

## 🎨 Personalização

### Adicionar Novos Gestos LIBRAS
1. Use `libras_data_collector.py` para coletar dados
2. Treine um novo modelo com TensorFlow
3. Atualize `labels.txt` com as novas classes

### Modificar Cores da Pintura
Edite `virtual_painter/VirtualPainter.py`:
```python
drawColor = (0, 255, 0)  # Verde
drawColor = (0, 0, 255)  # Vermelho
drawColor = (255, 0, 0)  # Azul
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🙏 Agradecimentos

- MediaPipe pela detecção de mãos
- OpenCV pela visão computacional
- TensorFlow pelo machine learning
- Comunidade Python pela documentação

---

**Desenvolvido com ❤️ para facilitar a comunicação por gestos**
"# Opencv" 
