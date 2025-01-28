import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

haar_cascade=cv.CascadeClassifier('haar_face.xml')

people=['Keira Knightley', 'Natalie Portman']

face_recognizer= cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')

def recognize_person():
    global img
    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect= haar_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=4)

    for (x, y, h, w) in faces_rect:
        faces_roi= gray[y:y+h, x:x+w]

        label, confidence= face_recognizer.predict(faces_roi)
        print(f'Label= {people[label]} with a confidence of {confidence}')

        cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        display_image()
        cv.imshow('Detected face', img)
        cv.waitKey(0)

def load_image():
    global img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv.imread(file_path)
        recognize_person()

def display_image():
    global img
    img_pil = Image.fromarray(img)
    img_display = ImageTk.PhotoImage(img_pil)
    image_label.config(image=img_display)
    image_label.image = img_display          

window = tk.Tk()
window.title("Keira Knightley or Natalie Portman")
window.geometry('700x580')
window.resizable(0, 0)

btn_load = tk.Button(window, text="Load image", command=load_image)
btn_load.place(x=300, y=500)
image_label= Label(window)

image_label.place(x=40, y=40)
btn_load.place(x=300, y=500)

window.mainloop()

