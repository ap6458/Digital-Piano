import cv2
from fer import FER

# Initialize FER detector (mtcnn=True = better face detection)
detector = FER(mtcnn=True)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect emotions
    results = detector.detect_emotions(frame)

    for face in results:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]

        # Get highest probability emotion
        emotion_label = max(emotions, key=emotions.get)
        confidence = emotions[emotion_label]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Put text
        text = f"{emotion_label} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,0,255), 2)

    cv2.imshow("FER Emotion Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()