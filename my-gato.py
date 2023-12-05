import cv2





cap = cv2.VideoCapture(0)       # default view for camera

while True:
    ret, frame = cap.read()         # gather frame from one camera
    if not ret:
        break

    cv2.imshow('logicam', frame)        # show the frame