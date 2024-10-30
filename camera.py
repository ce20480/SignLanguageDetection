import cv2 as cv
import numpy as np
import mediapipe as mp
from tensorflow import keras

cap = cv.VideoCapture(0)

# hand detection
mpHands = mp.solutions.hands
mpHandsMesh = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    output = mpHandsMesh.process(rgb)
    
    if output.multi_hand_landmarks:
        for i in output.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, i, mpHands.HAND_CONNECTIONS,
            landmark_drawing_spec=mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=5))
            print(i)
    
    
 
    # Display the resulting frame
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('frame', frame)   
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
