import cv2
import mediapipe as mp
import time
import os
from twilio.rest import Client
from dotenv import load_dotenv

# ----------------------------- Twilio Setup -----------------------------
load_dotenv()
ACCOUNT_SID = os.getenv("TWILIO_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_FROM")
TO_NUMBER = os.getenv("TWILIO_TO")

def send_emergency_sms():
    """Send an emergency SMS alert using Twilio"""
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 Emergency gesture detected! Please check immediately.",
            from_=FROM_NUMBER,
            to=TO_NUMBER
        )
        print("✅ Emergency SMS sent successfully!")
    except Exception as e:
        print("❌ Error sending SMS:", e)

# ----------------------------- Gesture Detection -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def is_thumb_tucked(hand_landmarks, hand_type):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return thumb_tip.x > index_finger_mcp.x if hand_type == "Right" else thumb_tip.x < index_finger_mcp.x

def is_fingers_outstretched(hand_landmarks):
    for i in [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]:
        if hand_landmarks.landmark[i].y > hand_landmarks.landmark[i - 1].y:
            return False
    return True

def is_fist(hand_landmarks, hand_type):
    if is_thumb_tucked(hand_landmarks, hand_type):
        for i in [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP,
        ]:
            if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y:
                return False
        return True
    return False

# ----------------------------- Main Loop -----------------------------
last_gesture = None
frame_counter = 0
max_frames_for_transition = 10

gesture_start_time = 0
gesture_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_type = "Right" if result.multi_handedness[idx].classification[0].label == "Right" else "Left"

            # Detect the sequence: open hand (thumb tucked) -> fist
            if is_fingers_outstretched(hand_landmarks) and is_thumb_tucked(hand_landmarks, hand_type):
                last_gesture = "Help part 1"
                frame_counter = 0
                gesture_start_time = time.time()
                gesture_detected = True

            elif last_gesture == "Help part 1" and is_fist(hand_landmarks, hand_type):
                if gesture_detected and (time.time() - gesture_start_time) >= 5:  # 5 seconds hold
                    print("Help sign detected and held for 5 seconds!")
                    send_emergency_sms()
                    gesture_detected = False
                    last_gesture = None
                    frame_counter = 0
                else:
                    print("Help sign shown but not held long enough.")

            else:
                if last_gesture == "Help part 1":
                    frame_counter += 1
                    if frame_counter > max_frames_for_transition:
                        last_gesture = None
                        gesture_detected = False
                        frame_counter = 0

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Emergency Gesture Detection", frame