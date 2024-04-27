import tkinter as tk 
from tkinter import messagebox
import cv2
import os
import numpy as np 
from PIL import Image

window = tk.Tk()
window.title("Face recognition system")

l1 = tk.Label(window, text="Name", font=("Algerian", 20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Age", font=("Algerian", 20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

l3 = tk.Label(window, text="Address", font=("Algerian", 20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)

def train_classifier():
    data_dir = "C:/Users/mayan/Downloads/Face Recognition/data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    
    for image in path:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(img)
        ids.append(id)
    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create() 
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result', 'Training dataset completed!')

b1 = tk.Button(window, text="Demo", font=("Algerian", 20), bg='orange', fg='red', command=train_classifier)
b1.grid(column=0, row=4)

def detect_face():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            id, pred = clf.predict(gray_image[y:y+h, x:x+w])
            confidence = int(100 * (1 - pred/300))

            if confidence > 77:
                if id == 1:
                    cv2.putText(img, "Mayank", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                elif id == 2:  
                    cv2.putText(img, "Anmol", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                elif id == 3:  
                    cv2.putText(img, "Komal", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                elif id == 4:  
                    cv2.putText(img, "kunal", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                elif id == 5:  
                    cv2.putText(img, "new member", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                else:
                    cv2.putText(img, "Unknown", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        return img
        def cleanup():
    # Release the camera


            if 'video_capture' in globals() and video_capture.isOpened():
                video_capture.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    # Quit the tkinter application
    window.quit()

    faceCascade = cv2.CascadeClassifier("lbpcascade_frontalface_opencv.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        cv2.imshow("face detection", img)
        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()

b2 = tk.Button(window, text="Detect The face", font=("Algerian", 20), bg='green', fg='white', command=detect_face)
b2.grid(column=1, row=4)

def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
    else:
        face_classifier = cv2.CascadeClassifier("lbpcascade_frontalface_opencv.xml")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            
            if len(faces) == 0:
                return None
            else:
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        id = 5
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
                    if cv2.waitKey(4) == 13 or img_id == 50:
                        break
                else:
                    cv2.imshow("Cropped face", frame)
                    if cv2.waitKey(4) == 13:
                        break
            else:
                print("Error: Failed to capture frame")
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generate dataset completed!')

b3 = tk.Button(window, text="Generate Dataset", font=("Algerian", 20), bg='pink', fg='black', command=generate_dataset)
b3.grid(column=2, row=4)

window.geometry("800x200")
window.mainloop()
cv2.destroyAllWindows()