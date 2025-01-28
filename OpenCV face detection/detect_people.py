import cv2 as cv

cap = cv.VideoCapture('videos/video4.mp4')

haar_cascade = cv.CascadeClassifier('haar_fullbody.xml')


while True:
    ret, frame= cap.read()
    frame=cv.resize(frame, None, fx=0.4, fy=0.4)
    detection=haar_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=3)

    for (x, y, w, h) in detection:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.imshow('Frame', frame)

    key=cv.waitKey(25)
    if key==27:
        break

cap.release()
cv.destroyAllWindows()