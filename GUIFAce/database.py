import tkinter as tk 
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np

window=tk.Tk()
window.title("Face recogination system")
l1=tk.Label(window,text="Name",font=("Algerian",20))
li.grid(column=0,row=0)
t1=tk.Entry(window,width=50,bd=5)
t1.grid(column=1,row=0)

l2=tk.Label(window,text="Name",font=("Algerian",20))
l2.grid(column=0,row=0)
t2=tk.Entry(window,width=50,bd=5)
t2.grid(column=1,row=0)

l3=tk.Label(window,text="Name",font=("Algerian",20))
l3.grid(column=0,row=0)
t3=tk.Entry(window,width=50,bd=5)
t3.grid(column=1,row=0)



window.geometery("800*200")
window.mainloop()
