import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotiondetector1.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)

labels = {
    0:'angry', 1:'disgust', 2:'fear',
    3:'happy', 4:'neutral', 5:'sad', 6:'surprise'
}

# For smoothing predictions
pred_list = []

while True:
    ret, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        image = gray[y:y+h, x:x+w]
        image = cv2.resize(image,(48,48))

        img = extract_features(image)

        pred = model.predict(img, verbose=0)[0]

        # ---- smoothing ----
        pred_list.append(pred)
        if len(pred_list) > 5:
            pred_list.pop(0)

        avg_pred = np.mean(pred_list, axis=0)

        # ---- get label ----
        label_index = np.argmax(avg_pred)
        prediction_label = labels[label_index]
        confidence = np.max(avg_pred)

        # ---- color based on emotion ----
        color = (0,255,0) if prediction_label == 'happy' else (0,0,255)

        # ---- better rectangle (thicker) ----
        cv2.rectangle(im, (x,y), (x+w,y+h), color, 3)

        # ---- background box for text (FER style) ----
        cv2.rectangle(im, (x, y-40), (x+w, y), color, -1)

        # ---- show label + confidence ----
        text = f"{prediction_label} ({confidence:.2f})"
        cv2.putText(im, text, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2)

    cv2.imshow("Emotion Detection", im)

    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
