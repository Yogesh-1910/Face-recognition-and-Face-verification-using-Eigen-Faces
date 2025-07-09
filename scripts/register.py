# scripts/register.py (Corrected with DNN Detector)

import cv2
import os
import argparse
import numpy as np

DNN_CONFIDENCE_THRESHOLD = 0.8 # Use a high confidence for good quality training data

def register_user(username, num_images=100):
    """Captures face images using a robust DNN detector."""
    output_dir = f"dataset/{username}"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load DNN Face Detector ---
    print("Loading DNN Face Detector for registration...")
    prototxt_path = os.path.join("face_detector", "deploy.prototxt.txt")
    weights_path = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        print("[ERROR] DNN model files not found. Please ensure 'face_detector' folder is in the project root.")
        return
        
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)

    cap = cv2.VideoCapture(0)
    count = 0
    print("\n--- Starting Face Capture (DNN Detector) ---")
    print("Press 'c' to capture. Press 'q' to quit.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret: break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        face_net.setInput(blob)
        detections = face_net.forward()

        # Find the largest, most confident face
        best_face_box = None
        max_confidence = 0
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > DNN_CONFIDENCE_THRESHOLD and confidence > max_confidence:
                max_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_face_box = box.astype("int")

        if best_face_box is not None:
            (startX, startY, endX, endY) = best_face_box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        
        status_text = f"Images Captured: {count}/{num_images}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Registration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and best_face_box is not None:
            (startX, startY, endX, endY) = best_face_box
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = gray[startY:endY, startX:endX]
            
            if face.size > 0:
                face_resized = cv2.resize(face, (100, 100))
                face_equalized = cv2.equalizeHist(face_resized)
                
                img_path = os.path.join(output_dir, f"{count}.jpg")
                cv2.imwrite(img_path, face_equalized)
                print(f"Saved {img_path}")
                count += 1
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a new user.")
    parser.add_argument("--name", required=True, help="Username for registration.")
    args = parser.parse_args()
    register_user(args.name)