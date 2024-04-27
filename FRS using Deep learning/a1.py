#This is the code for taking 100 pictures of each member
#  

import cv2


def generate_dataset():
    face_classifier = cv2.CascadeClassifier("lbpcascade_frontalface_opencv.xml")
    
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if faces is ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
    
    cap =cv2.VideoCapture(1)
    img_id=0

    while True:
        ret,frame =cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face =cv2.resize(face_cropped(frame),(200,200))
            face =cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path="data/"+"Mayank"+str(img_id)+".jpg"
            cv2.imwrite(file_name_path,face)
            cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            cv2.imshow("Cropped_Face",face)
            if cv2.waitKey(1)==13 or int(img_id)==100:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples is completed !!")

    generate_dataset()

import numpy as np
def my_label(image_name):
    name=image_name.split('.')[-3]
    if name=="Mayank":
        return np.array([1,0])
    elif name=="Manish":
        return np.array([0,1])



import os
from random import shuffle
from tqdm import tqdm 

def my_data():
    data=[]
    for img in tqdm(os.listdir("data")):
        path=os.path.join("data",img)
        img_data=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_data=cv2.resize(img_data(50,50))
        data.append([np.array(img_data),my_label(img)])
    shuffle(data)
    return data


data=my_data()

train=data[:2400]
test=data[2400:]
X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)

print(X_train.shape)
y_train=[i[i] for i in train]
X_test=np.array([i[0]for i in test]).reshape(-1,50,50,1)
print(x_test.shape)
y_test=[i[1]for i in test]

(2400,50,50,1)
(600,50,50,1)

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf 
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression


tf.reset_default_graph()
convnet=input_data(shape=[50,50,1])
convnet=conv_2d(convnet,32,5,activation='relu')

