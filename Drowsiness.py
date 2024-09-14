import cv2
import dlib
import numpy as np
import pygame
from imutils import face_utils
from scipy.spatial import distance
from datetime import datetime
import os
import pandas as pd
import time

# Initialize Pygame for alarm sound
pygame.mixer.init()
pygame.mixer.music.load('audio/alarm.mp3')

# User Input
user = input("Enter your name: ")

# Constants
EYE_ASPECT_RATIO_THRESHOLD = 0.3
YAWN_THRESHOLD = 20
ALARM_DURATION_THRESHOLD = 3  # Threshold for how long the event should last in seconds

# CSV file name and auto-creation
csv_file = "drowsiness_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, 'w') as f:
        f.write('User,Timestamp,Event,EAR\n')

# Load models
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


# Functions to calculate EAR and detect yawning
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_landmarks(image):
    rects = detector(image, 1)
    if len(rects) != 1:
        return None
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def top_lip(landmarks):
    return int(np.mean([landmarks[i, 1] for i in range(50, 53)] + [landmarks[i, 1] for i in range(61, 64)]))


def bottom_lip(landmarks):
    return int(np.mean([landmarks[i, 1] for i in range(65, 68)] + [landmarks[i, 1] for i in range(56, 59)]))


def detect_yawn(landmarks):
    if landmarks is None:
        return 0
    lip_dist = abs(top_lip(landmarks) - bottom_lip(landmarks))
    return lip_dist


def sound_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)


def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


# Logging function to write to CSV
def log_event(user, event, ear):
    now = datetime.now()
    log_data = pd.DataFrame({
        'User': [user],
        'Timestamp': [now.strftime("%Y-%m-%d %H:%M:%S")],
        'Event': [event],
        'EAR': [ear]
    })
    log_data.to_csv(csv_file, mode='a', header=False, index=False)


def draw_face_rectangles(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def draw_eye_contours(frame, shape):
    for (start, end) in [(lStart, lEnd), (rStart, rEnd)]:
        eye = shape[start:end]
        cv2.polylines(frame, [cv2.convexHull(eye)], True, (0, 255, 0), 1)


def draw_lips(frame, shape):
    # Landmarks for the lips
    top_lip_indices = list(range(48, 60))  # Top and bottom lip indices
    bottom_lip_indices = list(range(60, 68))

    # Extract points for the top and bottom lips
    top_lip_points = [shape[i] for i in top_lip_indices]
    bottom_lip_points = [shape[i] for i in bottom_lip_indices]

    # Convert to NumPy arrays
    top_lip_points = np.array(top_lip_points)
    bottom_lip_points = np.array(bottom_lip_points)

    # Draw contours for lips
    cv2.polylines(frame, [cv2.convexHull(top_lip_points)], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.polylines(frame, [cv2.convexHull(bottom_lip_points)], isClosed=True, color=(0, 255, 255), thickness=2)



# Track start times for yawning and eye closures
yawn_start_time = None
eye_closure_start_time = None

# Start video capture
video_capture = cv2.VideoCapture(0)
drowsy_logged = False
yawn_logged = False

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)
    draw_face_rectangles(frame, face_rectangle)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        draw_eye_contours(frame, shape)
        draw_lips(frame, shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Detect drowsiness (eyes closed for more than 3 seconds)
        if ear < EYE_ASPECT_RATIO_THRESHOLD:
            if eye_closure_start_time is None:
                eye_closure_start_time = time.time()  # Start timer

            # Check how long the eyes have been closed
            elapsed_time = time.time() - eye_closure_start_time
            if elapsed_time >= ALARM_DURATION_THRESHOLD:
                sound_alarm()
                cv2.putText(frame, "DROWSY", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                if not drowsy_logged:
                    log_event(user, "Drowsy (3+ seconds)", ear)
                    drowsy_logged = True
        else:
            eye_closure_start_time = None  # Reset timer if eyes open
            drowsy_logged = False
            stop_alarm()

        # Detect yawning (mouth open for more than 3 seconds)
        lip_dist = detect_yawn(shape)
        if lip_dist > YAWN_THRESHOLD:
            if yawn_start_time is None:
                yawn_start_time = time.time()  # Start timer

            # Check how long the mouth has been open
            elapsed_time = time.time() - yawn_start_time
            if elapsed_time >= ALARM_DURATION_THRESHOLD:
                sound_alarm()
                cv2.putText(frame, "Yawning", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                if not yawn_logged:
                    log_event(user, "Yawning (3+ seconds)", ear)
                    yawn_logged = True
        else:
            yawn_start_time = None  # Reset timer if mouth closes
            yawn_logged = False
            stop_alarm()

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
