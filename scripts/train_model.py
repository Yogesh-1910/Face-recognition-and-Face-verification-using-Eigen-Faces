# scripts/train_model.py

import numpy as np
import os
import cv2
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt

def train_model():
    dataset_path = "dataset/"
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset folder not found at '{os.path.abspath(dataset_path)}'.")
        print("Please ensure you are running this script from the project's root directory.")
        return

    print("Loading image dataset...")
    X, y, label_map = [], [], {}
    label_id = 0
    user_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    if len(user_folders) < 2:
        print(f"\n[ERROR] Training requires at least two registered users, but found only {len(user_folders)} in '{dataset_path}'.")
        return

    for user_dir in sorted(user_folders):
        user_path = os.path.join(dataset_path, user_dir)
        label_map[label_id] = user_dir
        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img.flatten())
                y.append(label_id)
        label_id += 1

    # --- ADDED THIS CRITICAL CHECK ---
    if not X:
        print(f"[ERROR] No images were loaded from the dataset folder: '{os.path.abspath(dataset_path)}'.")
        print("Please ensure the user subfolders contain valid image files.")
        return
    # --- END OF ADDED CHECK ---

    X = np.array(X)
    y = np.array(y)
    
    print("\nStep 1: Running PCA to analyze variance...")
    pca_analyzer = PCA(n_components=None, whiten=True).fit(X)

    plt.figure(figsize=(10, 7))
    plt.plot(np.cumsum(pca_analyzer.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by Number of Components")
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Explained Variance')
    plt.legend(loc='best')
    print("Displaying variance plot. Please close the plot window to continue training.")
    plt.show()

    try:
        n_components_optimal = np.where(np.cumsum(pca_analyzer.explained_variance_ratio_) >= 0.95)[0][0] + 1
    except IndexError:
        print("[WARNING] Could not automatically determine optimal components. Using 90% of available components.")
        n_components_optimal = int(len(X) * 0.9)

    print(f"\nStep 2: Optimal number of components to explain 95% of variance: {n_components_optimal}")

    print(f"Step 3: Re-training final PCA model with {n_components_optimal} components...")
    final_pca = PCA(n_components=n_components_optimal, whiten=True).fit(X)
    X_proj = final_pca.transform(X)

    model = {
        "pca": final_pca,
        "X_proj": X_proj,
        "y": y,
        "label_map": label_map
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/eigen_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\nModel training complete with optimal components!")
    print("Model saved to: models/eigen_model.pkl")

if __name__ == "__main__":
    train_model()