import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
eyes_glasses_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame(stream end?)")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (123, 255, 5), 3)
    faces = face_cascade.detectMultiScale(frame, scaleFactor= 1.1,minNeighbors= 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        frame_color = frame[y:y+h,x:x+w]

        eyes = eyes_cascade.detectMultiScale(frame_color, scaleFactor= 1.1, minNeighbors= 5)
        eyes_glasses = eyes_glasses_cascade.detectMultiScale(frame_color, scaleFactor= 1.1,
                                                             minNeighbors= 5)
        smile = smile_cascade.detectMultiScale(frame_color,scaleFactor= 1.1,minNeighbors= 5)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame_color, (ex,ey), (ex+ew,ey+eh), (0,0,255), 2)
        for(e_gx,e_gy,e_gw,e_gh) in eyes_glasses:
            cv2.rectangle(frame_color, (e_gx,e_gy), (e_gx + e_gw,e_gy + e_gh), (255,0,0), 2)
        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(frame_color, (sx,sy), (sx+sw,sy+sh), (0,255,0), 2)
    cv2.imshow("Video", frame)
    if cv2.waitKey(25) == ord("n"):
        break
cap.release()
cv2.destroyAllWindows()




