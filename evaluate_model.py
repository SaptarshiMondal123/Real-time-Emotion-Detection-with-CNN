import cv2
import numpy as np
from tensorflow.keras.models import load_model
from time import sleep

# Paths
CASCADE_PATH = 'your haarcascade classifier model'
MODEL_PATH = 'your trained model'

# Load Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier(CASCADE_PATH)

# Load emotion detection model
classifier = load_model(MODEL_PATH)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def detect_emotion(frame):
    global face_classifier, classifier, emotion_labels
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict emotion
        preds = classifier.predict(roi_gray)[0]
        emotion_idx = np.argmax(preds)

        # Determine emotion label
        if emotion_idx == 0:
            emotion_label = 'Angry'
        elif emotion_idx == 1:
            emotion_label = 'Disgust'
        elif emotion_idx == 2:
            emotion_label = 'Fear'
        elif emotion_idx == 3:
            emotion_label = 'Happy'
        elif emotion_idx == 4:
            emotion_label = 'Neutral'
        elif emotion_idx == 5:
            emotion_label = 'Sad'
        elif emotion_idx == 6:
            emotion_label = 'Surprise'
        else:
            emotion_label = 'Unknown'

        # Display emotion prediction
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            sleep(5)
            continue

        frame = detect_emotion(frame)

        cv2.imshow('Emotion Detection Model', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
