import cv2
import mediapipe as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Cores clean
CORSOFT = (255, 255, 255)
CORCONNECTION = (0, 200, 255)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    total_dedos = 0  # soma dos dedos de todas as mãos detectadas

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=CORSOFT, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=CORCONNECTION, thickness=2, circle_radius=2)
            )

            hand_label = hand_info.classification[0].label  # 'Left' ou 'Right'

            count = 0
            tip_ids = [8, 12, 16, 20]  # Indicador, médio, anelar, mínimo
            for tip_id in tip_ids:
                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                    count += 1

            # Polegar
            if hand_label == "Right":
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    count += 1
            else:  # Left hand
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    count += 1

            total_dedos += count  # soma da mão atual

    # Mostra total de dedos somados
    cv2.putText(frame, f'Dedos: {total_dedos}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.imshow("Contagem de Dedos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
