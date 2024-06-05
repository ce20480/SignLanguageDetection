import cv2 as cv
import mediapipe as mp

class HandDetector():
    def __init__(self, max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.5):
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
        
        
        # Check if the webcam is opened successfully
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

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
            
            # Display the resulting frame with hand landmarks
            cv.imshow('Hand Detection', frame)
            
            # Break the loop if 'q' key is pressed
            if cv.waitKey(1) == ord('q'):
                break

        # Release the video capture object and close all OpenCV windows
        self.cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    # Create an instance of HandDetector and start detecting hands
    hand_detector = HandDetector()
    hand_detector.detect_hands()
