# scripts/generate_final_report.py

import numpy as np
import os
import cv2
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- --------------------------------- ---
# ---           CONFIGURATION           ---
# --- --------------------------------- ---

# 1. Set the name of the model file you want to evaluate.
# MODEL_FILENAME = "eigen_model.pkl"          # For scikit-learn PCA model
MODEL_FILENAME = "eigen_model_numpy.pkl"    # For the NumPy-based model

# 2. IMPORTANT: Set this to the threshold you fine-tuned for your chosen model.
#    This value DRAMATICALLY affects the results.
# For scikit-learn model, this is usually low (e.g., 12.0 - 14.0)
# For NumPy model, this is usually high (e.g., 20000 - 25000)
DISTANCE_THRESHOLD = 21000

# 3. Define paths (these should be correct if you run from the project root)
TRAIN_DATA_PATH = "dataset/"
TEST_DATA_PATH = "test_data/"
REPORT_DIR = "reports"
# --- --------------------------------- ---
# ---         END CONFIGURATION         ---
# --- --------------------------------- ---


def load_images_and_labels_by_name(folder_path):
    """Loads images and uses folder names as string labels. This is a robust method."""
    X, y_names = [], []
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Directory not found: {os.path.abspath(folder_path)}")
        return np.array([]), np.array([])
        
    user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    for user_name in user_folders:
        user_path = os.path.join(folder_path, user_name)
        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img.flatten())
                y_names.append(user_name)
                
    return np.array(X), np.array(y_names)


def generate_final_report():
    """Generates a full performance report with multiple graphs."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    model_path = os.path.join("models", MODEL_FILENAME)
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Model not found at '{model_path}'. Please train the model first.")
        return

    print(f"--- Generating Report for: {MODEL_FILENAME} ---")

    # Unpack the trained model components
    is_numpy_model = 'eigenfaces' in model
    if is_numpy_model:
        mean_face, eigenfaces = model["mean_face"], model["eigenfaces"]
    else: # scikit-learn model
        recognizer = model.get("pca")
        if recognizer is None:
            print("[ERROR] The loaded model is not a valid scikit-learn PCA model.")
            return
        
    X_proj_train, y_train_ids, id_to_name_map = model["X_proj"], model["y"], model["label_map"]
    
    # Load test data
    X_test, y_true_names = load_images_and_labels_by_name(TEST_DATA_PATH)
    if len(X_test) == 0: return

    # Project test data into Eigenface space
    if is_numpy_model:
        X_test_proj = np.dot(X_test.astype(np.float32) - mean_face, eigenfaces.T)
    else:
        X_test_proj = recognizer.transform(X_test)
        
    # Get predictions, distances, and scores for all test images
    y_pred_names, distances_all = [], []
    for test_face_proj in X_test_proj:
        distances = np.linalg.norm(X_proj_train - test_face_proj, axis=1)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        distances_all.append(min_dist)
        
        if min_dist < DISTANCE_THRESHOLD:
            y_pred_names.append(id_to_name_map[y_train_ids[min_dist_idx]])
        else:
            y_pred_names.append("Unknown")

    # --- 1. Print Overall Metrics ---
    accuracy = accuracy_score(y_true_names, y_pred_names)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_names, y_pred_names, average='macro', zero_division=0)
    print("\n--- Overall Performance ---")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")

    # --- 2. Generate and Save Confusion Matrix ---
    print("\nGenerating Confusion Matrix...")
    all_names = sorted(list(set(y_true_names) | set(y_pred_names)))
    cm = confusion_matrix(y_true_names, y_pred_names, labels=all_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=all_names, yticklabels=all_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(REPORT_DIR, "1_confusion_matrix.png"))
    plt.close()
    print(f"Saved: {os.path.join(REPORT_DIR, '1_confusion_matrix.png')}")

    # --- 3. Generate and Save Accuracy vs. Components Graph ---
    print("\nGenerating Accuracy vs. Components graph...")
    X_train, y_train_names = load_images_and_labels_by_name(TRAIN_DATA_PATH)
    name_to_id_map_graph = {name: i for i, name in enumerate(sorted(list(set(y_train_names))))}
    y_train_ids_graph = np.array([name_to_id_map_graph[name] for name in y_train_names])
    id_to_name_map_graph = {v: k for k, v in name_to_id_map_graph.items()}
    component_range = list(range(1, 10)) + list(range(10, min(151, len(X_train)), 15))
    accuracies = []
    for n in component_range:
        if n >= len(X_train): break
        pca_temp = PCA(n_components=n).fit(X_train)
        X_train_proj_temp, X_test_proj_temp = pca_temp.transform(X_train), pca_temp.transform(X_test)
        y_pred_ids = [y_train_ids_graph[np.argmin(np.linalg.norm(X_train_proj_temp - face, axis=1))] for face in X_test_proj_temp]
        y_pred_names_temp = [id_to_name_map_graph[id] for id in y_pred_ids]
        accuracies.append(accuracy_score(y_true_names, y_pred_names_temp))
    plt.figure(figsize=(12, 7))
    plt.plot(component_range, accuracies, marker='o', linestyle='-')
    plt.xlabel("Number of Eigenfaces (Components)")
    plt.ylabel("Accuracy on Test Set")
    plt.title("Model Accuracy vs. Number of Components")
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, "2_accuracy_vs_components.png"))
    plt.close()
    print(f"Saved: {os.path.join(REPORT_DIR, '2_accuracy_vs_components.png')}")

    # --- 4. Generate and Save Distribution of Distances Graph ---
    print("\nGenerating Distribution of Distances graph...")
    correct_distances = [distances_all[i] for i in range(len(y_true_names)) if y_true_names[i] == y_pred_names[i]]
    incorrect_distances = [distances_all[i] for i in range(len(y_true_names)) if y_true_names[i] != y_pred_names[i]]
    plt.figure(figsize=(12, 7))
    sns.kdeplot(correct_distances, fill=True, color='g', label='Correct Matches')
    if incorrect_distances:
        sns.kdeplot(incorrect_distances, fill=True, color='r', label='Incorrect Matches')
    plt.axvline(x=DISTANCE_THRESHOLD, color='b', linestyle='--', label=f'Threshold ({DISTANCE_THRESHOLD})')
    plt.title('Distribution of Recognition Distances')
    plt.xlabel('Euclidean Distance in Eigenface Space')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, "3_distance_distribution.png"))
    plt.close()
    print(f"Saved: {os.path.join(REPORT_DIR, '3_distance_distribution.png')}")

    # --- 5. Generate and Save Precision-Recall Curve ---
    print("\nGenerating Precision-Recall Curve...")
    plt.figure(figsize=(10, 8))
    scores = DISTANCE_THRESHOLD - np.array(distances_all) # Invert distance to score
    for i, name in enumerate(sorted(id_to_name_map.values())):
        y_true_binary = (y_true_names == name)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, scores)
        pr_auc = auc(recall_curve, precision_curve)
        plt.plot(recall_curve, precision_curve, lw=2, label=f'PR curve for {name} (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for each Class')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(os.path.join(REPORT_DIR, "4_precision_recall_curve.png"))
    plt.close()
    print(f"Saved: {os.path.join(REPORT_DIR, '4_precision_recall_curve.png')}")

    print("\n\nReport generation complete! Check the 'reports' folder.")

if __name__ == "__main__":
    generate_final_report()