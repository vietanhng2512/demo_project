import cv2 as cv
# import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame(stream end?). Exiting...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame,cv.COLOR_HSV2BGR_FULL)
    # Display the resulting frame
    cv.imshow("Anh",gray)
    if cv.waitKey(25) == ord('n'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()