import cv2 as cv

#cap= cv.VideoCapture('videos/video3.mp4')
cap=cv.VideoCapture(0)

haar_cascade = cv.CascadeClassifier('haar_face.xml')


while True:
    ret, frame= cap.read()
    #frame=cv.resize(frame, None, fx=0.7, fy=0.7)
    gray_frame= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.7, minNeighbors=2)

    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.imshow('Frame', frame)

    key=cv.waitKey(10)
    if key==27:
        break

cap.release()
cv.destroyAllWindows()