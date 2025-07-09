# scripts/verify.py (with Verified Username Display)

import cv2
import numpy as np
import pickle
import argparse
import os

# --- CONFIGURATION ---
DISTANCE_THRESHOLD = 11.0
DNN_CONFIDENCE_THRESHOLD = 0.7

def verify_user(username_to_verify):
    """Verifies a user with their name and confidence score display."""
    # --- Load Eigenfaces Model ---
    try:
        # NOTE: This script works with the scikit-learn trained model 'eigen_model.pkl'
        # If you are using the NumPy model, change this to 'eigen_model_numpy.pkl'
        # and adjust the model loading logic accordingly.
        with open("models/eigen_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("\n[ERROR] Model not found. Please run train_model.py first.\n")
        return
        
    # This assumes the scikit-learn model structure
    model_pca = model["pca"]
    model_X_proj = model["X_proj"]
    model_y_labels = model["y"]
    model_label_map = model["label_map"]

    if username_to_verify not in model_label_map.values():
        print(f"[ERROR] Username '{username_to_verify}' not found in the trained model.")
        return

    # --- Load DNN Face Detector ---
    print("Loading DNN Face Detector...")
    prototxt_path = os.path.join("face_detector", "deploy.prototxt.txt")
    weights_path = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        print("[ERROR] DNN model files not found. Please ensure 'face_detector' folder is in the project root.")
        return
        
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)

    cap = cv2.VideoCapture(0)
    print(f"\n--- Verifying Identity for: {username_to_verify} (DNN Detector) ---")
    print("Look at the camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(0, detections.shape[2]):
            confidence_dnn = detections[0, 0, i, 2]

            if confidence_dnn > DNN_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face_x, face_y, face_w, face_h = startX, startY, endX - startX, endY - startY
                if face_w > 0 and face_h > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face = gray[face_y : face_y + face_h, face_x : face_x + face_w]

                    if face.size == 0: continue

                    face_resized = cv2.resize(face, (100, 100))
                    face_equalized = cv2.equalizeHist(face_resized)
                    face_proj = model_pca.transform([face_equalized.flatten()])[0]

                    distances = np.linalg.norm(model_X_proj - face_proj, axis=1)
                    min_dist_idx = np.argmin(distances)
                    min_dist = distances[min_dist_idx]
                    
                    best_match_label = model_y_labels[min_dist_idx]
                    best_match_name = model_label_map[best_match_label]
                    
                    if best_match_name == username_to_verify and min_dist < DISTANCE_THRESHOLD:
                        # --- THIS IS THE CORRECTED LINE ---
                        confidence_percent = 1.0 - (min_dist / DISTANCE_THRESHOLD)
                        text = f"Verified: {username_to_verify} ({confidence_percent:.0%})"
                        # --- END OF CORRECTION ---
                        
                        color = (0, 255, 0) # Green
                    else:
                        text = "Verification Failed"
                        color = (0, 0, 255) # Red

                    cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow(f"Verification for {username_to_verify}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a user's identity.")
    parser.add_argument("--name", required=True, help="The username to verify against.")
    args = parser.parse_args()
    verify_user(args.name)