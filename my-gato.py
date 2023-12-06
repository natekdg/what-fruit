import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

model = MobileNetV2(weights='imagenet')

def preprocess(frame):
    # resize to model input size
    frame_resized = cv2.resize(frame, (224, 224))

    # conver t to numpy array and process
    frame_array = np.array(frame_resized, dtype=np.float32)
    frame_preprocessed = preprocess_input(frame_array)

    return frame_preprocessed

cap = cv2.VideoCapture(0)       # use the default camera

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    processed_frame = preprocess(frame)

    # predict using preprocessed frame
    predictions = model.predict(np.expand_dims(processed_frame, axis=0))
    top_prediction = decode_predictions(predictions, top=1)[0][0]

    # display top of the frame
    cv2.putText(frame, top_prediction[1], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # show frame with prediction
    cv2.imshow('Webcam', frame)

    # close webcam if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release and close all cv2 windows
cap.release()
cv2.destroyAllWindows()
