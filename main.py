import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Define the connections manually to draw the hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def draw_landmarks(image, hand_landmarks):
    h, w, c = image.shape
    # Draw connections
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_pt = hand_landmarks[start_idx]
        end_pt = hand_landmarks[end_idx]
        
        cv2.line(image, 
                 (int(start_pt.x * w), int(start_pt.y * h)), 
                 (int(end_pt.x * w), int(end_pt.y * h)), 
                 (0, 255, 0), 2)
                 
    # Draw points
    for landmark in hand_landmarks:
        cv2.circle(image, (int(landmark.x * w), int(landmark.y * h)), 5, (0, 0, 255), cv2.FILLED)

def main():
    model_path = 'hand_landmarker.task'

    # Initialize MediaPipe Hand Tracking using the Tasks API
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Set up options
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2)

    print("Loading MediaPipe model...")
    with HandLandmarker.create_from_options(options) as landmarker:
        # Open Webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Starting webcam... Press 'q' to quit.")

        pTime = 0
        
        # Use perf_counter for robust monotonic timestamps
        start_time = time.perf_counter()

        try:
            while True:
                success, img = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # MediaPipe requires an RGB image
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
                
                # Calculate monotonic frame timestamp in ms
                frame_timestamp_ms = int((time.perf_counter() - start_time) * 1000)

                # Detect hand landmarks for the current video frame
                hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                # Draw the hand annotations on the original BGR image
                if hand_landmarker_result.hand_landmarks:
                    for hand_landmarks in hand_landmarker_result.hand_landmarks:
                        draw_landmarks(img, hand_landmarks)

                # Calculate and display FPS
                cTime = time.perf_counter()
                fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
                pTime = cTime

                cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

                # Display the resulting frame
                cv2.imshow("Hand Tracker Image", img)

                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
