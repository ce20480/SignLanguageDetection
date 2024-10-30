import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Create_Dataset import create_dataset
import mediapipe as mp
import cv2 as cv

data = 'C:\\Users\\Aship\\PycharmProjects\\SignLanguageDetection\\SignLanguageDetection\\data'
x_train, x_test, y_train, y_test, d = create_dataset(data)

## I need to extract the landmarks from xtrain and xtest - the landmarks will be the new xtrain and xtest
## I will use the mediapipe library to extract the landmarks

mpHands = mp.solutions.hands
mpHandsMesh = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

def get_coordinates(x):
    """
    Get the coordinates of all the landmarks of the hand.

    Parameters:
    x (numpy.ndarray): The frame to preprocess.

    Returns:
    numpy.ndarray: The preprocessed frame.
    """
    mpHands = mp.solutions.hands
    mpHandsMesh = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    h
    # Resize the frame to the target size
    frame_resized = cv.resize(x, (300, 300))
    rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
    output = mpHandsMesh.process(rgb)
    if output.multi_hand_landmarks:
        for i in output.multi_hand_landmarks:
            return i

x_train_landmarks = []
x_test_landmarks = []

for i in range(len(x_train)):
    x_train_landmarks.append(get_coordinates(x_train[i]))

for i in range(len(x_test)):
    x_test_landmarks.append(get_coordinates(x_test[i]))

x_train_landmarks = np.array(x_train_landmarks)
x_test_landmarks = np.array(x_test_landmarks)

print(x_train_landmarks.shape)
print(x_test_landmarks.shape)

# I will save the landmarks in a csv file
x_train_landmarks_df = pd.DataFrame(x_train_landmarks)
x_test_landmarks_df = pd.DataFrame(x_test_landmarks)

x_train_landmarks_df.to_csv('x_train_landmarks.csv', index=False)
x_test_landmarks_df.to_csv('x_test_landmarks.csv', index=False)

# I will save the labels in a csv file
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

y_train_df.to_csv('y_train.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)







