import cv2 as cv
import mediapipe as mp
import pyautogui
import numpy as np
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture Webcam Feed
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 60)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
screen_width, screen_height = pyautogui.size()

# Cursor Movement Variables
prev_x, prev_y = 0, 0
smooth_factor = 0.3
frame_skip = 2
frame_count = 0

def process_frame():
    global prev_x, prev_y, frame_count
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames for better performance

        frame = cv.flip(frame, 1)  # Mirror effect
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        height, width, _ = frame.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            # Only process the first detected hand (ignoring second hand)
            first_hand = results.multi_hand_landmarks[0]
            hand_label = results.multi_handedness[0].classification[0].label  # "Left" or "Right"

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, first_hand, mp_hands.HAND_CONNECTIONS)

            # Get Index Finger Tip Position
            landmarks = first_hand.landmark
            index_finger_tip = landmarks[8]
            x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

            # Map to Screen Coordinates
            screen_x = np.interp(x, [0, width], [0, screen_width])
            screen_y = np.interp(y, [0, height], [0, screen_height])

            # Apply Smoothing
            smoothed_x = prev_x + (screen_x - prev_x) * smooth_factor
            smoothed_y = prev_y + (screen_y - prev_y) * smooth_factor
            prev_x, prev_y = smoothed_x, smoothed_y
            pyautogui.moveTo(smoothed_x, smoothed_y, _pause=False)

            # Detect Pinch Gesture (Click)
            thumb_tip = landmarks[4]
            thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
            distance = np.hypot(x - thumb_x, y - thumb_y)
            if distance < 30:  
                pyautogui.click()

        # Display Frame
        cv.imshow("Gesture Control", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Run in a separate thread to improve performance
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()
thread.join()

cap.release()
cv.destroyAllWindows()
