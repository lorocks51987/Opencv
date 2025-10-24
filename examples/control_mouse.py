import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Resolução da tela
screen_width, screen_height = pyautogui.size()

# Inicializa câmera
camera = cv2.VideoCapture(1)

# Variáveis de suavização e clique
prev_x, prev_y = 0, 0
smoothening = 5  # quanto maior, mais suave
click_down = False

while True:
    ret, image = camera.read()
    if not ret:
        break

    image_height, image_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordenadas do dedo indicador e polegar
            x1 = int(hand_landmarks.landmark[8].x * image_width)  # indicador
            y1 = int(hand_landmarks.landmark[8].y * image_height)
            x2 = int(hand_landmarks.landmark[4].x * image_width)  # polegar
            y2 = int(hand_landmarks.landmark[4].y * image_height)

            # Círculos nos dedos
            cv2.circle(image, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 10, (0, 255, 255), cv2.FILLED)

            # Suavização do movimento
            mouse_x = prev_x + (screen_width / image_width * x1 - prev_x) / smoothening
            mouse_y = prev_y + (screen_height / image_height * y1 - prev_y) / smoothening
            pyautogui.moveTo(mouse_x, mouse_y)
            prev_x, prev_y = mouse_x, mouse_y

            # Distância entre dedos
            distance = math.hypot(x2 - x1, y2 - y1)

            # Clique quando os dedos se tocarem, com debounce
            if distance < 40:
                if not click_down:
                    pyautogui.click()
                    click_down = True
                    cv2.circle(image, ((x1 + x2)//2, (y1 + y2)//2), 15, (0,0,255), cv2.FILLED)
            else:
                click_down = False

    cv2.imshow("Controle de Mouse com Mão", image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

camera.release()
cv2.destroyAllWindows()
