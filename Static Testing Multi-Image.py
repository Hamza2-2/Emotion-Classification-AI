import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

new_model = load_model('C:/Users/ASUS/Music/SE Semester Project/Best_model_hamza.h5')

root = tk.Tk()
root.title("Emotion Classification Multi-Image")

uploaded_folder = None
results = []

def upload_folder():
    global uploaded_folder
    global results
    results.clear()  # Clear previous results
    result_display.config(state=tk.NORMAL)  # Enable result display for updating

    folder_path = filedialog.askdirectory()

    if folder_path:
        uploaded_folder = folder_path
        result_display.delete(1.0, tk.END)  # Clear the previous results from the text box
        result_display.insert(tk.END, f"Processing images in folder: {uploaded_folder}\n")
        result_display.insert(tk.END, "-"*50 + "\n")

def process_image(img_path):
    try:
        img = image.load_img(img_path)
        img_array = np.array(img)

        # Detect face
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            result_display.insert(tk.END, f"Face not detected in {os.path.basename(img_path)}\n")
        else:
            for x, y, w, h in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img_array[y:y+h, x:x+w]
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Detect face within the ROI
                facess = faceCascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey:ey + eh, ex:ex + ew]

            # Prepare the face image for prediction
            final_image = cv2.resize(face_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0
            Predictions = new_model.predict(final_image)
            predicted = np.argmax(Predictions)
            class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            # Display the predicted emotion class for each image
            result_display.insert(tk.END, f"Predicted Emotion for {os.path.basename(img_path)}: {class_names[predicted]}\n")
         
        result_display.insert(tk.END, "-"*50 + "\n")

    except Exception as e:
        result_display.insert(tk.END, f"Error processing {os.path.basename(img_path)}: {e}\n")
        result_display.insert(tk.END, "-"*50 + "\n")

    # Ensure that we continue with the next image even if no face is detected
    return

def predict_emotion():
    if uploaded_folder:
        file_list = [f for f in os.listdir(uploaded_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        for file_name in file_list:
            img_path = os.path.join(uploaded_folder, file_name)
            process_image(img_path)
        result_display.insert(tk.END, "Prediction completed for all images.\n")
    else:
        result_display.insert(tk.END, "Please upload a folder first.\n")

# Tkinter buttons and labels
upload_button = tk.Button(root, text="Upload Folder", command=upload_folder)
upload_button.pack(pady=10)

predict_button = tk.Button(root, text="Predict Emotion", command=predict_emotion)
predict_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=root.destroy, fg="white", bg="red", font=('Helvetica', 12))
exit_button.pack(pady=10)

# Text box to display results
result_display = tk.Text(root, height=15, width=70, font=('Helvetica', 12))
result_display.pack(padx=10, pady=10)
result_display.config(state=tk.DISABLED)  # Disable text box to prevent direct editing

root.mainloop()


