import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

locked = True

def unlock_condition(hand_landmarks):
    index_raised = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    other_fingers_folded = (
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        and hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
        and hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    )
    return index_raised and other_fingers_folded

def unlock_button_pressed():
    global locked
    locked = False
    print("Unlock button pressed")
    root.destroy()

def create_lock_ui():
    global root
    root = tk.Tk()
    root.title("Locked")
    root.geometry("300x200")

    unlock_button = tk.Button(root, text="Unlock", command=unlock_button_pressed)
    unlock_button.pack(pady=20)

    root.mainloop()

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            if not locked:
                ret, frame = cap.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    continue

                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

                frame.flags.writeable = False
                results = hands.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        if unlock_condition(hand_landmarks):
                            print("Unlock condition met!")
                            locked = False

                cv2.imshow("Hands", frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            else:
                create_lock_ui()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
