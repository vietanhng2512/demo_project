import cv2 as cv
# import numpy as np

cap = cv.VideoCapture("mt.mp4")
# i = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv.imshow("Original Image",frame)
    # cv.waitKey(0)
    # fileimg = "frame" + str(i) + ".jpg"
    # cv.imwrite(str(fileimg),frame)
    # i = i + 1
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame(stream end?). Exiting...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame,cv.COLOR_BGR2RGBA)
    # Display the resulting frame
    cv.imshow("Song",gray)
    if cv.waitKey(25) == ord('n'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()







