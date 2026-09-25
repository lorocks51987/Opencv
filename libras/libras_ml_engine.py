"""
=============================================================================
     MOTOR DE MACHINE LEARNING PARA LIBRAS (BASEADO EM LANDMARKS 3D)
=============================================================================
Usa normalização euclidiana invariante a escala, translação e iluminação.
Classificador K-Nearest Neighbors (KNN) implementado em NumPy puro.
Permite salvar/carregar dataset em JSON e gravar novas letras em tempo real.
=============================================================================
"""

import os
import json
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DATASET = os.path.join(SCRIPT_DIR, "libras_dataset.json")

# Dicionário de Instruções Educacionais Oficiais para o Alfabeto em LIBRAS
DICAS_EDUCACIONAIS = {
    "A": "Punho fechado com o polegar ereto encostado ao lado do indicador.",
    "B": "Quatro dedos esticados juntos para cima; polegar dobrado sobre a palma.",
    "C": "Mao curvada em formato de concha ou arco, imitando a letra 'C'.",
    "D": "Dedo indicador ereto para cima; pontas dos outros dedos tocam o polegar.",
    "E": "Dedos curvados para baixo com as unhas repousando sobre o polegar.",
    "F": "Indicador dobrado tocando o polegar (por FORA); outros 3 dedos abertos.",
    "G": "Indicador apontando para cima e polegar aberto reto na horizontal.",
    "H": "Indicador e medio para frente na horizontal com polegar entre eles.",
    "I": "Apenas o dedo mindinho levantado reto; todos os outros fechados.",
    "L": "Indicador e polegar esticados formando um angulo reto de 90 graus ('L').",
    "M": "Tres dedos (indicador, medio, anelar) voltados para baixo sobre o polegar.",
    "N": "Dois dedos (indicador e medio) voltados para baixo sobre o polegar.",
    "O": "Todos os dedos curvados tocando a ponta do polegar formando um 'O'.",
    "P": "Indicador na horizontal e medio para baixo com polegar apoiado.",
    "Q": "Indicador e polegar voltados para baixo formando uma pinca aberta.",
    "R": "Indicador e medio esticados para cima e CRUZADOS (um sobre o outro).",
    "S": "Punho totalmente fechado com o polegar passando POR CIMA dos dedos.",
    "T": "Indicador dobrado sobre o polegar (polegar fica por DENTRO); 3 abertos.",
    "U": "Indicador e medio esticados juntos, colados lado a lado para cima.",
    "V": "Indicador e medio esticados para cima e ABERTOS em formato de 'V'.",
    "W": "Tres dedos (indicador, medio e anelar) esticados e separados para cima.",
    "X": "Indicador dobrado em forma de gancho/anzol puxando para tras.",
    "Y": "Polegar e mindinho bem abertos (gesto havaiano / 'hang loose')."
}

def extrair_vetor_landmarks(lmList):
    """
    Extrai vetor de características invariante a translação e tamanho da mão:
    - Centraliza pelo pulso (ponto 0).
    - Normaliza pela distância do pulso ao nó da base do dedo médio (ponto 9).
    - Adiciona distâncias relativas entre as 5 pontas dos dedos [4, 8, 12, 16, 20].
    Retorna array NumPy 1D de 73 características.
    """
    if len(lmList) < 21:
        return None

    pts = np.array([[float(lm[1]), float(lm[2]), float(lm[3]) if len(lm) > 3 else 0.0] for lm in lmList])

    # 1. Centralização pelo pulso
    pulso = pts[0].copy()
    pts_centralizados = pts - pulso

    # 2. Escala euclidiana de referência (pulso ao nó do dedo médio)
    escala = np.linalg.norm(pts_centralizados[9])
    if escala < 1e-4:
        escala = 1.0

    pts_normalizados = pts_centralizados / escala

    # Vetor base das 21 coordenadas (x, y, z) normalizadas -> 63 features
    features = pts_normalizados.flatten().tolist()

    # 3. Distâncias-chave adicionais entre as pontas dos dedos
    # Pontas: Polegar(4), Indicador(8), Médio(12), Anelar(16), Mindinho(20)
    pontas = [4, 8, 12, 16, 20]
    for i in range(len(pontas)):
        for j in range(i + 1, len(pontas)):
            p1 = pts_normalizados[pontas[i]]
            p2 = pts_normalizados[pontas[j]]
            dist = float(np.linalg.norm(p1 - p2))
            features.append(dist)

    return np.array(features, dtype=np.float32)

class LibrasMLEngine:
    def __init__(self):
        self.dataset = {}  # { 'A': [vetor1, vetor2, ...], 'B': [...] }
        self.X_matrix = None
        self.y_labels = None
        self.carregar_ou_inicializar_dataset()
        self._atualizar_cache_matriz()

    def _atualizar_cache_matriz(self):
        """Prepara matrizes NumPy compactas para busca vetorial ultra-rápida (SIMD/BLAS)."""
        X_list = []
        y_list = []
        for classe, vetores in self.dataset.items():
            for v in vetores:
                X_list.append(v)
                y_list.append(classe)
        if X_list:
            self.X_matrix = np.array(X_list, dtype=np.float32)
            self.y_labels = np.array(y_list)
        else:
            self.X_matrix = None
            self.y_labels = None

    def carregar_ou_inicializar_dataset(self):
        """Carrega dataset salvo ou sintetiza referências canônicas das letras de LIBRAS."""
        if os.path.exists(FILE_DATASET):
            try:
                with open(FILE_DATASET, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.dataset = {k: [np.array(v, dtype=np.float32) for v in lista] for k, lista in raw.items()}
                print(f"[IA LIBRAS] Dataset carregado com {len(self.dataset)} letras cadastradas.")
                self._atualizar_cache_matriz()
                return
            except Exception as e:
                print(f"[AVISO] Erro ao ler dataset existente: {e}")

        # Dataset base inicial canônico
        self.dataset = self._gerar_dataset_sintetico_inicial()
        self.salvar_dataset()
        self._atualizar_cache_matriz()

    def salvar_dataset(self):
        """Persiste o dataset em JSON no disco com garantia de escrita."""
        serializavel = {k: [v.tolist() for v in lista] for k, lista in self.dataset.items()}
        try:
            with open(FILE_DATASET, "w", encoding="utf-8") as f:
                json.dump(serializavel, f, indent=2, ensure_ascii=False)
            print(f"[IA LIBRAS] Dataset salvo com sucesso no arquivo: '{FILE_DATASET}' ({len(self.dataset)} classes registradas).")
        except Exception as e:
            print(f"[ERRO AO SALVAR DATASET] {e}")
        self._atualizar_cache_matriz()

    def iniciar_gravacao_classe(self, letra):
        """Prepara gravação isolada em buffer sem destruir dados anteriores até concluir."""
        letra = letra.upper().strip()
        self._letra_ativa = letra
        self._buffer_gravacao = []

    def adicionar_amostra(self, letra, lmList):
        """Adiciona uma amostra da mão do usuário para uma letra no buffer de calibração."""
        letra = letra.upper().strip()
        vetor = extrair_vetor_landmarks(lmList)
        if vetor is None:
            return False

        if not hasattr(self, '_buffer_gravacao'):
            self._buffer_gravacao = []

        self._buffer_gravacao.append(vetor)
        return True

    def finalizar_gravacao_classe(self, letra):
        """Aplica a nova calibração gravada e salva imediatamente no disco."""
        letra = letra.upper().strip()
        if hasattr(self, '_buffer_gravacao') and len(self._buffer_gravacao) > 0:
            self.dataset[letra] = list(self._buffer_gravacao)
            self._buffer_gravacao = []
            self.salvar_dataset()
            return True
        return False

    def classificar(self, lmList, k=3):
        """
        Classifica os landmarks usando K-Nearest Neighbors (KNN) ultra-rápido vetorizado.
        Retorna: (letra_predita, confianca_0_a_100, dica_educacional)
        """
        vetor_atual = extrair_vetor_landmarks(lmList)
        if vetor_atual is None or self.X_matrix is None or len(self.X_matrix) == 0:
            return "---", 0.0, "Mostre a mao em frente a camera"

        # Distância euclidiana vetorizada para todos os exemplos em uma única chamada BLAS
        dists = np.linalg.norm(self.X_matrix - vetor_atual, axis=1)
        k_val = min(k, len(dists))
        indices_k = np.argpartition(dists, k_val - 1)[:k_val]
        indices_k = indices_k[np.argsort(dists[indices_k])]

        menor_distancia = float(dists[indices_k[0]])

        # Votação majoritária ponderada pela distância
        contagem = {}
        for idx in indices_k:
            cls = self.y_labels[idx]
            peso = 1.0 / (dists[idx] + 1e-4)
            contagem[cls] = contagem.get(cls, 0.0) + peso

        melhor_letra = max(contagem, key=contagem.get)

        # Cálculo de Confiança (menor distância = maior confiança)
        confianca = max(40.0, min(99.0, 100.0 - (menor_distancia * 45.0)))
        dica = DICAS_EDUCACIONAIS.get(melhor_letra, "Sinal identificado pela IA")

        return melhor_letra, confianca, dica

    def _gerar_dataset_sintetico_inicial(self):
        """
        Gera poses canônicas tridimensionais para as principais letras de LIBRAS
        para que o sistema já funcione de saída antes de qualquer calibração.
        """
        dataset = {}
        # Constrói 21 marcos com base nas regras canônicas de LIBRAS
        letras_iniciais = ["A", "B", "C", "D", "E", "F", "G", "I", "L", "O", "R", "S", "U", "V", "W", "Y"]

        for l in letras_iniciais:
            dataset[l] = []
            pts_simulados = self._sintetizar_pose_letra(l)
            if pts_simulados:
                v = extrair_vetor_landmarks(pts_simulados)
                if v is not None:
                    # Adiciona com pequenas perturbações para robustez
                    dataset[l].append(v)
                    for _ in range(4):
                        ruido = np.random.normal(0, 0.02, v.shape).astype(np.float32)
                        dataset[l].append(v + ruido)

        return dataset

    def _sintetizar_pose_letra(self, letra):
        """Cria coordenadas 21x3 representando a anatomia da letra."""
        # 0 = Pulso
        base = [[i, 0, 0, 0] for i in range(21)]
        base[0] = [0, 300, 450, 0]
        base[9] = [9, 300, 320, 0]  # junta do dedo médio

        # Posições das juntas MCP
        base[1] = [1, 340, 410, 0]; base[2] = [2, 360, 380, 0]; base[3] = [3, 370, 350, 0]
        base[5] = [5, 330, 330, 0]; base[6] = [6, 330, 280, 0]; base[7] = [7, 330, 240, 0]
        base[10] = [10, 300, 270, 0]; base[11] = [11, 300, 230, 0]
        base[13] = [13, 270, 340, 0]; base[14] = [14, 270, 290, 0]; base[15] = [15, 270, 250, 0]
        base[17] = [17, 240, 360, 0]; base[18] = [18, 240, 310, 0]; base[19] = [19, 240, 270, 0]

        # Pontas esticadas (Y menor) vs dobradas (Y maior)
        if letra == "D":
            # Indicador ereto para cima (8), outros dobrados tocando o polegar
            base[8] = [8, 330, 160, 0]    # Indicador para cima
            base[4] = [4, 320, 300, 0]    # Polegar tocando base dos dedos
            base[12] = [12, 310, 305, 0]  # Médio dobrado tocando polegar
            base[16] = [16, 280, 315, 0]  # Anelar dobrado
            base[20] = [20, 250, 330, 0]  # Mindinho dobrado
        elif letra == "I":
            # Mindinho ereto para cima (20), outros dobrados com polegar sobre os dedos
            base[20] = [20, 240, 180, 0]  # Mindinho esticado
            base[4] = [4, 320, 340, 0]    # Polegar travando os dedos
            base[8] = [8, 320, 320, 0]    # Indicador dobrado
            base[12] = [12, 300, 320, 0]  # Médio dobrado
            base[16] = [16, 280, 320, 0]  # Anelar dobrado
        elif letra == "Y":
            # Polegar (4) e Mindinho (20) eretos e bem abertos
            base[4] = [4, 400, 320, 0]    # Polegar bem aberto para a direita
            base[20] = [20, 210, 190, 0]  # Mindinho bem aberto para a esquerda
            base[8] = [8, 320, 320, 0]    # Indicador dobrado
            base[12] = [12, 300, 320, 0]  # Médio dobrado
            base[16] = [16, 280, 320, 0]  # Anelar dobrado
        elif letra == "L":
            # Indicador (8) e Polegar (4) eretos em 90 graus
            base[8] = [8, 330, 160, 0]    # Indicador para cima
            base[4] = [4, 410, 360, 0]    # Polegar aberto em 90 graus
            base[12] = [12, 300, 320, 0]; base[16] = [16, 280, 320, 0]; base[20] = [20, 250, 330, 0]
        elif letra == "V":
            # Indicador e Médio eretos e ABERTOS
            base[8] = [8, 350, 170, 0]
            base[12] = [12, 280, 170, 0]
            base[4] = [4, 320, 340, 0]; base[16] = [16, 280, 320, 0]; base[20] = [20, 250, 330, 0]
        elif letra == "U":
            # Indicador e Médio eretos e JUNTOS
            base[8] = [8, 315, 170, 0]
            base[12] = [12, 300, 170, 0]
            base[4] = [4, 320, 340, 0]; base[16] = [16, 280, 320, 0]; base[20] = [20, 250, 330, 0]
        elif letra == "W":
            # Três dedos eretos (8, 12, 16)
            base[8] = [8, 350, 170, 0]; base[12] = [12, 305, 160, 0]; base[16] = [16, 265, 170, 0]
            base[4] = [4, 320, 340, 0]; base[20] = [20, 250, 330, 0]
        elif letra == "B":
            # Quatro dedos eretos juntos
            base[8] = [8, 335, 170, 0]; base[12] = [12, 305, 160, 0]; base[16] = [16, 275, 170, 0]; base[20] = [20, 245, 190, 0]
            base[4] = [4, 315, 340, 0]
        elif letra == "A":
            # Punho fechado, polegar encostado na lateral do indicador
            base[4] = [4, 355, 300, 0]
            base[8] = [8, 325, 315, 0]; base[12] = [12, 300, 315, 0]; base[16] = [16, 275, 315, 0]; base[20] = [20, 250, 325, 0]
        elif letra == "E":
            # Dedos curvados com pontas tocando o polegar
            base[4] = [4, 310, 330, 0]
            base[8] = [8, 325, 290, 0]; base[12] = [12, 300, 290, 0]; base[16] = [16, 275, 290, 0]; base[20] = [20, 250, 300, 0]
        elif letra == "C":
            # Dedos curvados em arco C
            base[4] = [4, 360, 320, 0]
            base[8] = [8, 330, 220, 0]; base[12] = [12, 300, 220, 0]; base[16] = [16, 270, 230, 0]; base[20] = [20, 245, 250, 0]
        elif letra == "O":
            # Todos os dedos tocam a ponta do polegar em círculo
            base[4] = [4, 320, 280, 0]
            base[8] = [8, 320, 275, 0]; base[12] = [12, 310, 275, 0]; base[16] = [16, 300, 280, 0]; base[20] = [20, 290, 285, 0]
        else:
            base[8] = [8, 330, 160, 0]

        return base
