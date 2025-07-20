from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import cv2

# Load the model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into the new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

# Setting image resizing parameters
WIDTH = 48
HEIGHT = 48
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the video
cap = cv2.VideoCapture('video_test.mp4')  # Change 'video.mp4' to your video file path
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Predicting the emotion
        yhat = loaded_model.predict(cropped_img)
        emotion = labels[int(np.argmax(yhat))]
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        print("Emotion: " + emotion)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()

