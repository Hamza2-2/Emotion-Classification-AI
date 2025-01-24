import os
import tensorflow as tf
import cv2
import numpy as np
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

new_model = tf.keras.models.load_model('C:/Users/ASUS/Music/SE Semester Project/Best_model_hamza.h5')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_indicator(img, percentage):
    def percentage_to_color(p):
        return (0, int(255 * p), int(255 * (1 - p)))

    levels = 10
    indicator_width = 80
    indicator_height = 220
    level_width = indicator_width - 20
    level_height = int((indicator_height - 20) / levels - 5)


    img_levels = int(percentage * levels)
    cv2.rectangle(img, (10, img.shape[0] - (indicator_height + 10)), 
                  (10 + indicator_width, img.shape[0] - 10), (0, 0, 0), cv2.FILLED)

    for i in range(img_levels):
        level_y_b = int(img.shape[0] - (20 + i * (level_height + 5)))
        cv2.rectangle(img, (20, level_y_b - level_height), 
                      (20 + level_width, level_y_b), percentage_to_color(i / levels), cv2.FILLED)

  
    text = f"{int(percentage * 100)}%"  
    text_position = (10, img.shape[0] - (indicator_height + 20))  
    cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cap = cv2.VideoCapture(0)
face_roi = None

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        font_scale=1.5
        font=cv2.FONT_HERSHEY_PLAIN
        status = "No face detected"
        x1, y1, w1, h1 = 0, 0, 225, 75
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
    else:
        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = roi_color

        if face_roi is not None:
            final_image = cv2.resize(face_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0

            predictions = new_model.predict(final_image)
            emotion_index = np.argmax(predictions)
            emotion_confidence = predictions[0][emotion_index]

            emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
            status = emotions[emotion_index]
            if(status!="Neutral"):
                draw_indicator(frame, emotion_confidence)
      
            font_scale=1.5
            font=cv2.FONT_HERSHEY_PLAIN

            if (emotion_index == 0):
                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

            elif (emotion_index == 1):
                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

            elif (emotion_index == 2):
                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0,0))

            elif (emotion_index == 3):
                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
                cv2.putText(frame,status,(100,150),font,3,(0,255,0),2,cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0))

            elif (emotion_index == 4):
                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
                cv2.putText(frame,status,(100,150),font,3,(0,255,0),2,cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0))

            elif (emotion_index == 5):
                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

            elif (emotion_index == 6):
               
                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame,status,(100,150),font,3,(255,0,0),2,cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
            

    cv2.imshow('Face Emotion Recognition', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
