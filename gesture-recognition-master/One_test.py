import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['action']  # 하나의 모션만 인식합니다.
seq_length = 30

model = load_model('new_model.h5')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10,11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            v = v.reshape(-1, 3 * 20)  # 형상 변경: 20x3 -> 1x60
            v = v.flatten()  # 형상 변경: 1x60 -> 60
            v_expanded = np.pad(v, (0, 99 - len(v)), 'constant')

            seq.append(v)
            if len(seq) < seq_length:
                continue
            else:
                seq = np.array(seq[-seq_length:], dtype=np.float32)
                pred = model.predict(seq[np.newaxis, ...])[0]
                detected_action = actions[np.argmax(pred)]

                if detected_action == 'action':  # 예측된 행동이 우리가 찾는 행동과 일치하는지 확인합니다.
                    print("Detected action!")
                else:
                    print("No action detected.")

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', img)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
