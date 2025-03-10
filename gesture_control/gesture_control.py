import cv2 as cv
import mediapipe as mp
import pyautogui
import numpy as np
import threading
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

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

# Click Control Variables
click_cooldown = 0.5  # Time in seconds between clicks
last_click_time = 0
click_active = False  # Track if a click is currently being held

# Swipe Gesture Variables
swipe_threshold = 0.05  # Minimum horizontal movement to trigger a swipe
prev_mid_x = None  # Track previous midpoint for swipe detection

# Screenshot Gesture Variables
screenshot_cooldown = 1.0  # Time in seconds between screenshots
last_screenshot_time = 0

def is_finger_extended(landmark_tip, landmark_dip):
    """Returns True if the finger is extended (tip above DIP joint)."""
    return landmark_tip.y < landmark_dip.y

def is_finger_folded(landmark_tip, landmark_dip):
    """Returns True if the finger is folded (tip below DIP joint)."""
    return landmark_tip.y > landmark_dip.y

def detect_thumb_gesture(landmarks):
    """Detects thumbs-up or thumbs-down while ensuring other fingers are folded."""
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Check if all fingers except the thumb are folded
    if (is_finger_folded(index_tip, landmarks[6]) and
        is_finger_folded(middle_tip, landmarks[10]) and
        is_finger_folded(ring_tip, landmarks[14]) and
        is_finger_folded(pinky_tip, landmarks[18])):

        # Check if the thumb is significantly above or below the MCP joint
        if thumb_tip.y > thumb_mcp.y:  # Thumb is above MCP → Thumbs Up
            return "up"
        elif thumb_tip.y < thumb_mcp.y:  # Thumb is below MCP → Thumbs Down
            return "down"

    return None  # No valid gesture detected

def detect_peace_gesture(landmarks):
    """Detects the 'peace' gesture (index and middle fingers extended, others folded)."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Check if index and middle fingers are extended, others are folded
    if (is_finger_extended(index_tip, landmarks[6]) and  # Index finger extended
       is_finger_extended(middle_tip, landmarks[10]) and  # Middle finger extended
       is_finger_folded(thumb_tip, landmarks[2]) and  # Thumb folded
       is_finger_folded(ring_tip, landmarks[14]) and   # Ring finger folded
       is_finger_folded(pinky_tip, landmarks[18])):    # Pinky finger folded

        return True

    return False


def detect_swipe_gesture(landmarks):
    """Detects swipe left or right using index and middle fingers."""
    global prev_mid_x

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_dip = landmarks[6]
    middle_dip = landmarks[10]

    # Check if index and middle fingers are extended, others are folded
    if (is_finger_extended(index_tip, index_dip) and
        is_finger_extended(middle_tip, middle_dip) and
        is_finger_folded(ring_tip, landmarks[14]) and
        is_finger_folded(pinky_tip, landmarks[18])):

        # Calculate midpoint between index and middle fingertips
        mid_x = (index_tip.x + middle_tip.x) / 2

        if prev_mid_x is not None:
            # Calculate horizontal movement
            horizontal_movement = mid_x - prev_mid_x

            if horizontal_movement > swipe_threshold:  # Swipe Right
                prev_mid_x = None  # Reset after detecting a swipe
                return "right"
            elif horizontal_movement < -swipe_threshold:  # Swipe Left
                prev_mid_x = None  # Reset after detecting a swipe
                return "left"

        prev_mid_x = mid_x  # Update previous midpoint

    return None

def detect_click(landmarks, width, height):
    """Detects pinch gesture for click, ensuring it happens only once per pinch."""
    global last_click_time, click_active

    index_tip = landmarks[8]
    thumb_tip = landmarks[4]

    x, y = int(index_tip.x * width), int(index_tip.y * height)
    thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)

    distance = np.hypot(x - thumb_x, y - thumb_y)

    if distance < 30:  # If fingers are close enough to click
        if not click_active and time.time() - last_click_time > click_cooldown:
            pyautogui.click()
            print("Click")  # Debugging
            last_click_time = time.time()
            click_active = True  # Set click as active
    else:
        click_active = False  # Reset click when fingers move apart

def process_frame():
    global prev_x, prev_y, frame_count, last_screenshot_time
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

        if results.multi_hand_landmarks:
            first_hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, first_hand, mp_hands.HAND_CONNECTIONS)

            landmarks = first_hand.landmark

            # Detect Strict Gestures
            thumb_gesture = detect_thumb_gesture(landmarks)
            swipe_gesture = detect_swipe_gesture(landmarks)
            is_fist = detect_peace_gesture(landmarks)

            # Gesture Prioritization
            if thumb_gesture == "up":
                pyautogui.scroll(50)  # Scroll Up
                print("Scrolling Up")  # Debugging
            elif thumb_gesture == "down":
                pyautogui.scroll(-50)  # Scroll Down
                print("Scrolling Down")  # Debugging
            elif swipe_gesture == "left":
                pyautogui.press('left')  # Simulate Left Arrow Key Press
                print("Swiping Left")  # Debugging
            elif swipe_gesture == "right":
                pyautogui.press('right')  # Simulate Right Arrow Key Press
                print("Swiping Right")  # Debugging
            elif is_fist and time.time() - last_screenshot_time > screenshot_cooldown:
                # Take a screenshot
                screenshot = pyautogui.screenshot()
                screenshot.save("screenshot.png")
                print("Screenshot Taken")  # Debugging
                last_screenshot_time = time.time()
            else:
                # Cursor Movement (Only if no other gestures are detected)
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

            # Check Click Gesture Separately
            detect_click(landmarks, width, height)

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

