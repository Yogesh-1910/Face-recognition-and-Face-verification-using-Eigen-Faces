# 🔐 Secure Face Login System using Eigenfaces (PCA)

This project is a complete, real-time face recognition system built in Python. It demonstrates the classic **Eigenfaces** algorithm for both **1-to-N Identification** ("Who is this person?") and **1-to-1 Verification** ("Is this person who they claim to be?").

The system uses a modern Deep Neural Network (DNN) for highly accurate face detection, feeding clean, well-aligned face images to the Eigenfaces recognition engine. It also includes robust pre-processing steps like Histogram Equalization to handle varied lighting conditions.

---

## ✨ Key Features

- **Real-time Face Identification:** Identifies a person from a database of known users via webcam.
- **Real-time Face Verification:** Verifies if a person matches a specific claimed identity.
- **Robust Face Detection:** Uses a pre-trained DNN model for accurate and stable face detection, minimizing errors from poor lighting or pose.
- **Data Pre-processing:** Implements Histogram Equalization to normalize images and improve recognition accuracy under different lighting conditions.
- **Multiple Training Methods:** Includes two training scripts:
  - `train_model.py`: Using the standard `scikit-learn` PCA library.
  - `train_numpy.py`: A "from scratch" implementation using NumPy's SVD for a deeper understanding of the algorithm.
- **Performance Evaluation:** A comprehensive script (`generate_report.py`) to calculate metrics like Accuracy, Precision, Recall and generate performance graphs like the Confusion Matrix.

---

## 🛠️ Technology Stack

- **Language:** Python 3.x
- **Core Libraries:**
  - [OpenCV](https://opencv.org/): For image processing, webcam access, and the DNN module.
  - [NumPy](https://numpy.org/): For all numerical operations and matrix manipulations.
  - [Scikit-learn](https://scikit-learn.org/): For PCA implementation and performance metrics.
- **Plotting & Reporting:**
  - [Matplotlib](https://matplotlib.org/): For generating graphs.
  - [Seaborn](https://seaborn.pydata.org/): For creating the aesthetic confusion matrix.

---

## 📁 Project Structure

```
secure-face-login/
├── dataset/               # Stores training images (80% of data)
├── test_data/             # Stores testing images (20% of data)
├── models/                # Stores the trained .pkl model files
├── reports/               # Stores the generated performance graphs
├── face_detector/         # Contains the DNN face detector model files
├── scripts/
│   ├── register.py        # Enroll new users
│   ├── train_model.py     # Train using scikit-learn's PCA
│   ├── identify.py        # Perform 1-to-N identification
│   ├── verify.py          # Perform 1-to-1 verification
│   └── generate_report.py # Evaluate performance and create graphs
├── venv/                  # Python virtual environment
└── requirements.txt
```

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/secure-face-login.git
cd secure-face-login
```

### 2. Create and Activate a Virtual Environment

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content:

```
opencv-python
numpy
scikit-learn
matplotlib
seaborn
```

Then install using:

```bash
pip install -r requirements.txt
```

### 4. Download the Face Detector Model

Create a folder named `face_detector` in your project root and download these two files into it:

- **Model Definition**: `deploy.prototxt.txt`
- **Model Weights**: `res10_300x300_ssd_iter_140000.caffemodel`

You can download them from:

- [deploy.prototxt.txt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
- [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel)

---

## 📖 How to Use the System

### Step 1: Register Users

Use the `--name` argument (no spaces). Capture around 100 images per user.

```bash
python scripts/register.py --name person 1
python scripts/register.py --name person 2
```

---

### Step 2: Split Data and Train the Model

Manually split data into an 80/20 ratio. Move 80 images to `dataset/` and 20 to `test_data/`.

Then choose one of the training methods:

```bash
# Option A: scikit-learn PCA
python scripts/train_model.py

# Option B: NumPy SVD implementation
python scripts/train_numpy.py
```

---

### Step 3: Perform Identification or Verification

Set an appropriate `DISTANCE_THRESHOLD` inside the script.

```bash
# Identification (1-to-N)
python scripts/identify.py

# Verification (1-to-1)
python scripts/verify.py --name person1
```

---

### Step 4: Evaluate Performance

Make sure test data and model are ready. Then:

```bash
python scripts/generate_report.py
```

Check the `reports/` folder for confusion matrix and metrics.

---

## 🧠 Core Concepts Explained

- **Eigenfaces (PCA):** Reduces dimensionality of face images by extracting the most significant features (principal components).
- **DNN Face Detector:** A pre-trained deep learning model to detect faces accurately under varied conditions.
- **Identification vs. Verification:**
  - *Identification*: 1-to-N matching — "Who is this person?"
  - *Verification*: 1-to-1 matching — "Is this person really User X?"

---

## 📈 Results

- **Accuracy, Precision, Recall**: Achieved **100%** on test set.
- **No Misclassifications**: Clean confusion matrix due to:
  - Robust DNN detection
  - Histogram Equalization
  - Well-tuned threshold

---

## 🔮 Future Improvements

- **Liveness Detection**: Detect blinks to avoid spoofing via photos.
- **GUI**: Build a front-end using Tkinter or PyQt.
- **Advanced Models**: Integrate FaceNet or ArcFace for better performance on difficult datasets.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more info.
