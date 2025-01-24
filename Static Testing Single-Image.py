import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import os

# Disable TensorFlow optimizations and logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the pre-trained model
new_model = load_model('C:/Users/ASUS/Music/SE Semester Project/Best_model_hamza.h5')

# Initialize the Tkinter root window
root = tk.Tk()
root.title("Hamza")

# Global variable to store the uploaded image
uploaded_image = None

def upload_image():
    global uploaded_image

    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )

    if file_path:
        # Load the selected image
        uploaded_image = image.load_img(file_path)
        # Display the uploaded image using matplotlib
        plt.imshow(uploaded_image)
        plt.title("Uploaded Image")
        plt.show()

def predict_emotion():
    if uploaded_image:
        # Load the face detection model
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert PIL image to numpy array
        img_array = np.array(uploaded_image)
        # Convert the image to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            print("Face not detected")
        else:
            for x, y, w, h in faces:
                # Extract the region of interest (ROI) for the face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img_array[y:y+h, x:x+w]
                # Draw a rectangle around the detected face
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Detect face within the ROI
                facess = faceCascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey:ey + eh, ex:ex + ew]

            # Display the image with the detected face
            plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            plt.show()

            # Preprocess the face ROI for prediction
            final_image = cv2.resize(face_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0
            # Predict the emotion using the pre-trained model
            Predictions = new_model.predict(final_image)
            predicted = np.argmax(Predictions)

            # Display the predicted emotion class
            result_label.config(text=f"Predicted Emotion Class: {predicted}")
    else:
        result_label.config(text="Please upload an image first.")

# Create and pack the upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=(10, 15))

# Create and pack the predict button
predict_button = tk.Button(root, text="Predict Emotion", command=predict_emotion)
predict_button.pack(pady=(10, 15))

# Create and pack the result label
result_label = tk.Label(root, text="Predicted Emotion Class: None", font=('Helvetica', 14))
result_label.pack()

# Create and pack the exit button
exit_button = tk.Button(root, text="Exit", command=root.destroy, fg="white", bg="red", font=('Helvetica', 12))
exit_button.pack(pady=10)

# Start the Tkinter main loop
root.mainloop()