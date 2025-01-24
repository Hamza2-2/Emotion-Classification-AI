import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

new_model = load_model('C:/Users/ASUS/Music/SE Semester Project/Best_model_hamza.h5')

root = tk.Tk()
root.title("Hamza")

uploaded_image = None

def upload_image():
    global uploaded_image

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )

    if file_path:
        uploaded_image = image.load_img(file_path)          
        plt.imshow(uploaded_image)
        plt.title("Uploaded Image")
        plt.show()

panda = None

def predict_emotion():
    if uploaded_image:
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert PIL image to numpy array
        img_array = np.array(uploaded_image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            print("Face not detected")
        else:
            for x, y, w, h in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img_array[y:y+h, x:x+w]
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Detect face within the ROI
                facess = faceCascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey:ey + eh, ex:ex + ew]

            plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            plt.show()

            final_image = cv2.resize(face_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0
            Predictions = new_model.predict(final_image)
            predicted= np.argmax(Predictions)

            # Display the predicted emotion class
            result_label.config(text=f"Predicted Emotion Class: {predicted}")
    else:
        result_label.config(text="Please upload an image first.")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=(10, 15))

predict_button = tk.Button(root, text="Predict Emotion", command=predict_emotion)
predict_button.pack(pady=(10, 15))

result_label = tk.Label(root, text="Predicted Emotion Class: None", font=('Helvetica', 14))
result_label.pack()
exit_button = tk.Button(root, text="Exit", command=root.destroy, fg="white", bg="red", font=('Helvetica', 12))
exit_button.pack(pady=10)
root.mainloop()
