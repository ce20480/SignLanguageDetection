import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# LANDMARKS METHOD

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.5, model_path='SignLanguageDetection\\SignLanguageCNN.h5', hand="Right"):
        """
        Initialize the hand detector with given parameters.

        Parameters:
        max_num_hands (int): Maximum number of hands to detect.
        min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand detection.
        min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand landmarks tracking.
        model_path (str): Path to the pre-trained CNN model.
        hand (str): Specifies which hand to detect ("Right" or "Left").
        """
        # Initialize the video capture object for the webcam
        self.cap = cv.VideoCapture(0)

        # Initialize MediaPipe Hands module
        self.mpHands = mp.solutions.hands
        self.mpHandsMesh = self.mpHands.Hands(static_image_mode=False, 
                                              max_num_hands=max_num_hands, 
                                              min_detection_confidence=min_detection_confidence, 
                                              min_tracking_confidence=min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        
        # Load the pre-trained CNN model
        self.model = load_model(model_path)
        self.hand = hand
        
        # Check if the webcam is opened successfully
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

    def get_landmarks(self, frame):
        # landmarks are stored as shape: (21, 3)
        # where each row is (x, y, z)
        landmarks = []
        results = self.mpHandsMesh.process(frame)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
        return np.array(landmarks), results

    def predict(self, landmarks):
        """
        Predict the sign language gesture from the landmarks.

        Parameters:
        landmarks (numpy.ndarray): Normalized landmarks from the bounding box.

        Returns:
        str: Predicted sign language gesture.
        """
        # Ensure landmarks are in the correct shape (21, 3)
        if landmarks.shape != (21, 3):
            return None

        # Reshape landmarks for prediction
        landmarks = landmarks.reshape(1, 21, 3)

        # Predict the sign language gesture
        prediction = self.model.predict(landmarks)

        # Get the predicted sign language gesture
        predicted_gesture = np.argmax(prediction)

        # Return the predicted sign language gesture
        return predicted_gesture
    
    def draw_bounding_box(self, frame, hand_landmarks, margin=20):
        """
        Draw a bounding box around the detected hand with an added margin.

        Parameters:
        frame (numpy.ndarray): Frame from the webcam.
        hand_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Hand landmarks detected by MediaPipe.
        margin (int): Margin to add to the bounding box.
        """
        h, w, _ = frame.shape
        x_min = w
        y_min = h
        x_max = 0
        y_max = 0

        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y
        
        # Add margin to the bounding box
        x_min = max(x_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_max = min(x_max + margin, w)
        y_max = min(y_max + margin, h)

        cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Normalize landmarks to fit within the bounding box
        normalized_landmarks = []
        for lm in hand_landmarks.landmark:
            normalized_x = (lm.x * w - x_min) / (x_max - x_min)
            normalized_y = (lm.y * h - y_min) / (y_max - y_min)
            normalized_landmarks.append((normalized_x, normalized_y, lm.z))
        
        return np.array(normalized_landmarks)

    def live_video(self):
        """
        Display the live video feed from the webcam.
        """
        while True:
            try:
                # Read the frame from the webcam
                success, frame = self.cap.read()
                if not success:
                    break

                # Convert the BGR frame to RGB
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Detect hands in the frame
                landmarks, results = self.get_landmarks(rgb_frame)

                # Draw the landmarks and bounding box on the frame
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                        normalized_landmarks = self.draw_bounding_box(frame, hand_landmarks, margin=30)

                        # Predict the sign language gesture using the normalized landmarks
                        predicted_gesture = self.predict(normalized_landmarks)

                        # Display the predicted sign language gesture on bottom left corner
                        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
                        if predicted_gesture is not None:
                            cv.putText(frame, labels[predicted_gesture], (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display fps on the frame
                fps = self.cap.get(cv.CAP_PROP_FPS)
                cv.putText(frame, "FPS: " + str(int(fps)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the frame
                cv.imshow("Hand Detection", frame)

                # Break the loop when 'q' is pressed
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"An error occurred: {e}")

        # Release the webcam and close all windows
        self.cap.release()
        cv.destroyAllWindows()

def main():
    # Initialize the HandDetector object
    hand_detector = HandDetector()

    # Display the live video feed from the webcam
    hand_detector.live_video()

if __name__ == "__main__":
    main()