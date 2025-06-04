import cv2
import os
import mediapipe as mp

def capture_gesture_images():
    gesture_name = input("Enter the gesture name: ").strip()
    if not gesture_name:
        print("Gesture name cannot be empty!")
        return
    
    save_path = os.path.join("asl_dataset", gesture_name)
    os.makedirs(save_path, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    count = 0
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image!")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            img_path = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
            print(f"Saved: {img_path}")
        
        cv2.putText(frame, f"Capturing {gesture_name}: {count}/100", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Capture", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Gesture '{gesture_name}' images collected successfully!")

if __name__ == "__main__":
    capture_gesture_images()
