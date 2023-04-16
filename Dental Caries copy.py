import cv2
import numpy as np
import os
from tkinter import filedialog
from tkinter import *
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
m=Tk(className='Dental')
def f():
    X_train = []
    path = "D:\\done"
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key = cv2.contourArea)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        features = cv2.mean(gray, mask = mask)       
        X_train.append(features)
    kmeans = KMeans(n_clusters=2, n_init=1)
    kmeans.fit(X_train)
    X_test = []
    path=filedialog.askopenfilename()
    path=path.replace('/','\\') 
    print(path)
    print("1 for healthy and 0 for decayed")
    img = cv2.imread(path)
    l=Label(m,text="Selected Image Is : ")
    l.place(x=25,y=105)
    fr=Frame(m)
    fr.place(x=25,y=145)
    im=ImageTk.PhotoImage(Image.open(path).resize((174, 157), Image.LANCZOS))
    l1=Label(fr,image=im)
    l1.image=im
    l1.pack()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    

    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = cv2.contourArea)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)    
    features = cv2.mean(gray, mask = mask)    
    X_test.append(features)
    i = cv2.imread(path)
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 50, 200, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(i, (x, y), (x + w, y + h), (0, 0, 255), 2)
    fr1=Frame(m)
    fr1.place(x=300,y=80)
    i1=Image.fromarray(i).resize((174, 157), Image.LANCZOS)
    im1=ImageTk.PhotoImage(image=i1)
    l5=Label(fr1,image=im1)
    l5.image=im1
    l5.pack()
    y_pred = kmeans.predict(X_test)
    print(y_pred)
    print(type(y_pred))
    if(y_pred[0]==1):
        l6=Label(m,text='Healthy')
        l6.place(x=360,y=260)
    else:
        l6=Label(m,text='Caries Found !')
        l6.place(x=360,y=260)
    y_true = np.array([1])
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(accuracy*100)
l3=Label(m,text="Dental Caries Prediction System",font="Bold")
l3.place(relx=0.5, rely=0.1, anchor=CENTER)
l2=Label(m,text="Click here to upload : ")
l2.place(x=25,y=50)    
but=Button(m,text='Upload',command=f,foreground='Green')
but.place(x=25,y=80)
l4=Label(m,text='Result : ')
l4.place(x=325,y=50)
m.geometry("500x360")
m.mainloop()