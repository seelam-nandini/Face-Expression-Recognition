import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

# Loading the model structure
json_file = open("faceemotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("faceemotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # Loading the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0) #Initializing Web Cam
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Main loop to process the video input
while True:
    ret, frame = webcam.read()
    break  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the captured frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale image

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_gray = cv2.resize(roi_gray, (48, 48)) 
        img = extract_features(roi_gray)
        pred = model.predict(img) 
        prediction_label = labels[pred.argmax()]
        cv2.putText(frame, prediction_label, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Emotion Detector', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
