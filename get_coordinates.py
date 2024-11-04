import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Create_Dataset import create_dataset
import mediapipe as mp
import cv2 

# data = 'C:\\Users\\Aship\\PycharmProjects\\SignLanguageDetection\\SignLanguageDetection\\data'
# x_train, x_test, y_train, y_test, d = create_dataset(data)

## I need to extract the landmarks from xtrain and xtest - the landmarks will be the new xtrain and xtest
## I will use the mediapipe library to extract the landmarks

def get_coords(image):
    # image = cv2.imread(image)
    mpHands = mp.solutions.hands
    mpDraw = mp.solutions.drawing_utils
    hands = mpHands.Hands(static_image_mode=True, max_num_hands=1,min_detection_confidence=0.000001)
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    results = hands.process(image)
    coords = []
    if not results.multi_hand_landmarks:
        print("not working")
        coords = [0]*63
        plt.imshow(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            print(hand_landmarks)
            mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)
            plt.imshow(image)
            for landmark in hand_landmarks.landmark:
                coords.extend([landmark.x, landmark.y,landmark.z])
    return coords

# x_train_landmarks = []
# x_test_landmarks = []

# for i in range(len(x_train)):
#     x_train_landmarks.append(get_coordinates(x_train[i]))

# for i in range(len(x_test)):
#     x_test_landmarks.append(get_coordinates(x_test[i]))

# x_train_landmarks = np.array(x_train_landmarks)
# x_test_landmarks = np.array(x_test_landmarks)

# print(x_train_landmarks.shape)
# print(x_test_landmarks.shape)

# # I will save the landmarks in a csv file
# x_train_landmarks_df = pd.DataFrame(x_train_landmarks)
# x_test_landmarks_df = pd.DataFrame(x_test_landmarks)

# x_train_landmarks_df.to_csv('x_train_landmarks.csv', index=False)
# x_test_landmarks_df.to_csv('x_test_landmarks.csv', index=False)

# # I will save the labels in a csv file
# y_train_df = pd.DataFrame(y_train)
# y_test_df = pd.DataFrame(y_test)

# y_train_df.to_csv('y_train.csv', index=False)
# y_test_df.to_csv('y_test.csv', index=False)







