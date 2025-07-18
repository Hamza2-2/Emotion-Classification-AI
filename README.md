# Emotion-Classification-AI
This project implements a deep learning-based Emotion Classification AI System using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It can classify human emotions in both real-time via webcam and from static images. Ideal for integrating into smart AI systems, education tools, or emotion-aware applications.

## 🧠 Model Overview
- Framework: TensorFlow / Keras

- Model Type: Convolutional Neural Network (CNN)

- Output Classes:

  - Angry

  - Disgust

  - Fear

  - Happy

  - Sad

  - Surprise

  - Neutral

Model Format: .h5 file (Keras format)

## 📁 Repository Structure
Emotion-Classification-AI/

Emotion-Classification-AI/
├── AI Model Training.py              # Train the CNN model using a labeled dataset
├── Emotion AI Model.h5              # Pre-trained CNN model file
├── Real-Time Testing.py             # Real-time webcam-based emotion detection
├── Static Testing Multi-Image.py    # Predict emotion for multiple static images
├── Static Testing Single-Image.py   # Predict emotion for a single static image
└── README.md           

## 🧠 Model Overview

- Framework: **TensorFlow / Keras**
- Model: **Convolutional Neural Network (CNN)**
- Output Classes: `['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`
- Trained model format: `.h5`

 
## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/Hamza2-2/Emotion-Classification-AI.git
cd Emotion-Classification-AI

```
### 2. Install Required Libraries
Ensure Python 3.10+ is installed, then run:
 ```
pip install tensorflow opencv-python matplotlib numpy
```

## 🚀 How to Use

🔴 Real-Time Emotion Detection
 ```
python "Real-Time Testing.py"
```

🟡 Static Image Testing

Single Image
```
python "Static Testing Single-Image.py
```

Multiple Images
```
python "Static Testing Multi-Image.py"
```
🧪 Model Training
To train the model from scratch:
```
python "AI Model Training.py"
```
Ensure you modify the script to point to your dataset directory.


## Screenshots

<img width="340" height="606" alt="image" src="https://github.com/user-attachments/assets/d81c2b1b-41b5-4666-8846-96e91e132f71" />

<img width="576" height="413" alt="image" src="https://github.com/user-attachments/assets/fd5533f7-aa5c-484d-8373-2f929fa1b62c" />


## 📌 Notes
Pre-trained model is saved as: Emotion AI Model.h5

Modify script paths if running from a different working directory.  

## 👨‍💻 Developer

Hamza Afzal

BSCS, Bahria University, Islamabad

GitHub: Hamza2-2
