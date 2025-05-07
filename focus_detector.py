
import cv2
import time
import numpy as np
import mediapipe as mp
from plyer import notification
import pyttsx3

engine = pyttsx3.init()

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)


mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14]


focus_score = 100
start_time = None
is_focus_time = True
mobile_start_time = None
no_face_start = None
focus_start = None
break_start = None

def speak_alert(text):
    engine.say(text)
    engine.runAndWait()


def check_mobile_usage(frame):
    global mobile_start_time

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            nose_tip = landmarks[1]  
            if nose_tip.y > 0.6:
                if mobile_start_time is None:
                    mobile_start_time = time.time()  
                elif time.time() - mobile_start_time >= 300:  
                    speak_alert("Mobile usage detected! You've been looking down for 5 minutes.")
                    mobile_start_time = None  # Reset 
            else:
                mobile_start_time = None  # Reset 

    return frame


def calculate_ear(landmarks, eye_indices):
    # Compute Eye Aspect Ratio (EAR)
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])

    # distances
    vertical1 = np.linalg.norm(p2 - p4)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p6)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def detect_focus(frame, focus_score):
    global no_face_start, mobile_start_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        # No face detected
        if no_face_start is None:
            no_face_start = time.time()
        elif time.time() - no_face_start >= 300:  # 5 minutes
            notification.notify(title="No Face Detected",
                                message="No face detected for 5 minutes.",
                                timeout=5)
            speak_alert("No face detected for 5 minutes.")
            no_face_start = None
        focus_score = max(0, focus_score - 0.01)  # no face
        return frame, focus_score
    else:
        no_face_start = None  # Reset 

    for face_landmarks in results.multi_face_landmarks:
        # Draw face landmarks
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_spec)

        # Calculate EAR for both eyes
        landmarks = face_landmarks.landmark
        left_ear = calculate_ear(landmarks, LEFT_EYE)
        right_ear = calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        # Drowsiness detection
        if avg_ear < 0.25:
            focus_score = max(focus_score - 0.1, 0)  
            cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Distraction detection based on head orientation 
        nose_tip = landmarks[1]
        if nose_tip.x < 0.3 or nose_tip.x > 0.7:
            focus_score = max(focus_score - 0.2, 0)  
            cv2.putText(frame, "DISTRACTED", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Yawning detection based on mouth opening
        top_lip = landmarks[MOUTH[0]]
        bottom_lip = landmarks[MOUTH[1]]
        mouth_open = np.linalg.norm(
            np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
        if mouth_open > 0.05:
            focus_score = max(focus_score - 0.1, 0)  # Decrease focus score
            cv2.putText(frame, "YAWNING DETECTED", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Phone usage detection based on head position
        if nose_tip.y > 0.6:
            if mobile_start_time is None:
                mobile_start_time = time.time()
            elif time.time() - mobile_start_time >= 300:
                notification.notify(title="Phone Usage Detected",
                                    message="You've been looking down for 5 minutes.",
                                    timeout=5)
                speak_alert("You've been looking down for 5 minutes.")
                mobile_start_time = None
        else:
            mobile_start_time = None

    # Increase focus score 
    if avg_ear > 0.3 and (0.3 <= nose_tip.x <= 0.7):  
        focus_score = min(focus_score + 0.5, 100) 
        cv2.putText(frame, "FOCUSING", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Focus alert
    if focus_score < 20:
        notification.notify(title="Focus Warning",
                            message="Your focus level is low!",
                            timeout=5)
        speak_alert("Your focus level is low!")

    return frame, focus_score



def manage_timer(focus_minutes, break_minutes):
    global is_focus_time, focus_start, break_start

    now = time.time()
    focus_sec = focus_minutes * 60
    break_sec = break_minutes * 60

    if is_focus_time:
        if focus_start is None:
            focus_start = now
        elapsed = now - focus_start
        remaining = max(0, int(focus_sec - elapsed))
        mins, secs = divmod(remaining, 60)
        if remaining == 0:
            notification.notify(title="Focus Time Over",
                                message="Time for a break!", timeout=5)
            speak_alert("Focus time is over. Time for a break!")
            is_focus_time = False
            focus_start = None
            break_start = now
        return "Focus", f"{mins:02d}:{secs:02d}", "--:--"
    else:
        if break_start is None:
            break_start = now
        elapsed = now - break_start
        remaining = max(0, int(break_sec - elapsed))
        mins, secs = divmod(remaining, 60)
        if remaining == 0:
            notification.notify(title="Break Over",
                                message="Back to work!", timeout=5)
            speak_alert("Break is over. Back to work!")
            is_focus_time = True
            break_start = None
            focus_start = now
        return "Break", "--:--", f"{mins:02d}:{secs:02d}"
