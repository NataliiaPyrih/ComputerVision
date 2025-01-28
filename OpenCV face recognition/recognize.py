import os
import cv2 as cv
import numpy as np

people=['Keira Knightley', 'Natalie Portman']
DIR= 'Faces'

features=[]
labels=[]

haar_cascade= cv.CascadeClassifier('haar_face.xml')

def get_faces():
    for person in people:
        path=os.path.join(DIR, person)
        label= people.index(person)

        for img in os.listdir(path):
            img_path= os.path.join(path, img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect= haar_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi= gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

get_faces()
print(f'Face detection is finished. Number of faces is {len(features)}')

face_recognizer= cv.face.LBPHFaceRecognizer_create()

features= np.array(features, dtype='object')
labels= np.array(labels)

face_recognizer.train(features, labels)
print(f'Training is finished!')

face_recognizer.save('face_recognizer.yml')


            
