#part 4
import tkinter as tk 
from tkinter import messagebox
import cv2
from PIL import Image
import numpy as np 

window=tk.Tk()
window .title("Face recognition system")
l1=tk.Label(window,text="Name",font=("Algerian",20))
l1.grid(column=0,row=0)
t1=tk.Entry(window,width=50,bd=5)
t1.grid(column=1,row=0)

l2=tk.Label(window,text="Age",font=("Algerian",20))
l2.grid(column=0,row=0)
t2=tk.Entry(window,width=50,bd=5)
t2.grid(column=1,row=0)

l3=tk.Label(window,text="Address",font=("Algerian",20))
l3.grid(column=0,row=0)
t3=tk.Entry(window,width=50,bd=5)
t3.grid(column=1,row=0)
def train_classifier(data_dir):
    data_dir="C:/Users/mayan/Downloads/Face Recognition/data"
    path=[os.path.join(data_dir,f)for f in os.listdir(data_dir)]
    faces=[]
    ids=[]
    
    for image in path:
        img= Image.open(image).convert('L');
        imageNp=np.array(img,'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids=np.array(ids)

    #   Train the classfier and save
    clf = cv2.face.LBPHFaceRecognizer_create() 
    clf.train(faces,ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result','Training dataset complerted!')
   

b1=tk.Button(window,text="Training",font=("Algerian",20),bg='orange',fg='red',command=train_classifier)
b1.grid(column=0,row=4)

def detect_face():
    def draw_boundary(image,classifier,scaleFactor,minNeighbours,color,text,clf):
        gray_image =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray_image,scale,scaleFactor,minNeighbors)

    coords =[]
    for(x,y,w,h) in features:
        cv2.reactangle(img(x,y),(x+w,y+h),color,2)
        id,pred=clf.predict(gray_image[y:y+w,x+w])
        confidence=int(100*(1-pred/300))

        if confidence>77:
            if id==1:
                cv2.putText(img,"Mayank",(x,y-5),cv2.Font_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
            if id==2:
                cv2.putText(img,"Kunal",(x,y-5),cv2.Font_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)

            else:
                cv2.putText(img,"Unknown",(x,y-5),cv2.Font_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)
            coords=[x,y,w,h]
        return coords
    def recognize(img,clf,faceCascade):
        coords=draw_boundary(img,faceCascade,1.1,10,(255,255,255),"Face",clf)
        return img

    faceCascade =cv2.CascadeClassifier("lbpcascade_frontalface_opencv.xml")
    clf=cv2.face.LBPHFaceRecognizer_create()

    clf.read("classifier.xml")
    video_capture = cv2.VidelCapture(0)
    while True:
        ret,img=video_capture.read()
        img=recognize(img,clf,faceCascade)
        cv2.imshow("face detection",img)
        if cv2.waitkey(1)==13:
            break
    video_capture.release()
    cv2.destroyAllWindows()


b2=tk.Button(window,text="Detect The face",font=("Algerian",20),bg='green',fg='white',command=detect_face)
b2.grid(column=1,row=4)
def generate_dataset():

    if (t1.get()=="" or t2.get()=="" or t3.get()==""):
        messagebox.showinfo('Result','Please provide complete details of the user')
    else:
        face_classifier = cv2.CascadeClassifier("lbpcascade_frontalface_opencv.xml")
    
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        if len(faces) == 0:
            return None
        else:
            # Get the largest face (assuming it's the closest to the camera)
            max_area = 0
            for (x, y, w, h) in faces:
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_face = (x, y, w, h)
            x, y, w, h = largest_face
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width to 1920 pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height to 1080 pixels
    
    id = 1
    img_id = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cropped_face = face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"data/user.{id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped face", face)
                if cv2.waitKey(4) == 13 or img_id == 10:
                    break
            else:
                cv2.imshow("Cropped face", frame)  # Show full frame if no face is detected
                if cv2.waitKey(4) == 13:
                    break
        else:
            print("Error: Failed to capture frame")
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo('result','generate f=dataset completed!')
    


b3=tk.Button(window,text="Generate Dataset",font=("Algerian",20),bg='pink',fg='black',command=generate_dataset)
b3.grid(column=2,row=4)


window.geometry("800x200")
window.mainloop()