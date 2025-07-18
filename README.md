# Emotion-Classification-AI
This project implements an **Emotion Classification AI System** using a Convolutional Neural Network (CNN) trained with TensorFlow/Keras. It supports both static image testing and real-time emotion recognition via webcam.

---

## 📁 Repository Structure
Emotion-Classification-AI/

├── AI Model Training.py # Script to train the emotion classifier

├── Emotion AI Model.h5 # Trained CNN model file

├── Real-Time Testing.py # Real-time webcam emotion detection

├── Static Testing Multi-Image.py # Test on multiple static images

├── Static Testing Single-Image.py # Test on a single static image

└── README.md # Project documentation

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

🚀 How to Use

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

📌 Notes
Pre-trained model is saved as: Emotion AI Model.h5

Modify script paths if running from a different working directory.  
