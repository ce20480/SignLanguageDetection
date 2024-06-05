import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle

class HandDetector():
    def __init__(self, max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.5, model_path='early_model.sav', hand="Right"):
        """
        Initialize the hand detector with given parameters.

        Parameters:
        max_num_hands (int): Maximum number of hands to detect.
        min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand detection.
        min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand landmarks tracking.
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
        
        self.model = pickle.load(open(model_path, 'rb'))
        self.hand = hand
        
        # Check if the webcam is opened successfully
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")
        
    def preprocess_frame(self, frame, target_size=(300,300)):
        """
        Preprocess the frame for the model input.
        
        Parameters:
        frame (numpy.ndarray): The frame to preprocess.
        target_size (tuple): The target size of the frame.
        
        Returns:
        numpy.ndarray: The preprocessed frame.
        """
        
        frame_resized = cv.resize(frame, target_size)
        frame_normalized = frame_resized / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=0)
        return frame_expanded
    

    def detect_hands(self):
        """
        Detect hands in the video stream and display the results.
        """
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert the frame to RGB as MediaPipe processes images in RGB format
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process the RGB frame for hand landmarks
            output = self.mpHandsMesh.process(rgb)
            
            # If hand landmarks are found, draw them on the frame
            if output.multi_hand_landmarks:
                for hand_landmarks in output.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS,
                                               landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=5))
                    
            # Preprocessed frame for the model input
            frame_preprocessed = self.preprocess_frame(frame)
            
            # Predict the gesture using the model
            prediction = self.model.predict(frame_preprocessed)
            predicted_label = np.argmax(prediction, axis=1)[0]
        
                        
            # Display the predicted gesture on the frame on the bottom left corner
            label_text = "Predicted Gesture: " + str(predicted_label)
            cv.putText(frame, label_text, (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            
            # Display the resulting frame with hand landmarks
            cv.imshow('Hand Detection', frame)
            
            # Break the loop if 'q' key is pressed
            if cv.waitKey(1) == ord('q'):
                break

        # Release the video capture object and close all OpenCV windows
        self.cap.release()
        cv.destroyAllWindows()
        
    def landmark_coordinates(self, hand_id=0):
        """Get the coordinates of all the landmarks of the hand.

        Args:
            hand_id (int, optional): The id of the hand to get the landmarks. Defaults to 0.
            
        Returns:
            dict: A dictionary containing the landmark id as key and the landmark coordinates as value.
        """
        
        landmark_coords = {}
        if self.output.multi_hand_landmarks:
            hand = self.output.multi_hand_landmarks[hand_id]
            for id, landmark in enumerate(hand.landmark):
                landmark_coords[id] = (landmark.x, landmark.y, landmark.z)
                if self.hand == "Left":
                    landmark_coords[id][0] *= -1
        return landmark_coords
    
def main():
    # Create an instance of HandDetector and start detecting hands
    hand_detector = HandDetector()
    hand_detector.detect_hands()

if __name__ == "__main__":
    # Create an instance of HandDetector and start detecting hands
    hand_detector = HandDetector()
    hand_detector.detect_hands()
