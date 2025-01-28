import cv2 as cv

image= cv.imread('people2.jpg')
image= cv.resize(image, None, fx=0.2, fy=0.2)
#cv.imshow('Women', image)

gray_image= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray women', gray_image)

haar_cascade= cv.CascadeClassifier('haar_face.xml')
faces_rect= haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=1)
print(f'Number of faces found is {len(faces_rect)}')
for (x, y, w, h) in faces_rect:
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected face', image)
cv.waitKey(0)