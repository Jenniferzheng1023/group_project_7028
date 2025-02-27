
import tkinter as tk
import tkinter.filedialog
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from PIL import Image, ImageTk
from torchvision import transforms as transforms
from test import main, model, model_cycleGAN
from data import create_dataset
from models import create_model
import cv2 as cv
import os
import torch
import argparse

# 创建UI
win = tk.Tk()
win.title("picture process")
win.geometry("1280x1080")

# 声明全局变量
original = Image.new('RGB', (300, 400))
save_img = Image.new('RGB', (300, 400))
count = 0
e2 = None
e2 = str(e2)
file_name = None
img2 = tk.Label(win)

def choose_file():
	'''选择一张照片'''
	select_file = tk.filedialog.askopenfilename(title='select the picture')
	global file_name
	file_name=select_file
	e.set(select_file)
	load = Image.open(select_file)
	load = transforms.Resize((400,400))(load)
	# 声明全局变量
	global original
	original = load
	render = ImageTk.PhotoImage(load)
	img  = tkinter.Label(win,image=render)
	img.image = render
	img.place(x=100,y=100)

def edge_detect():
    '''边缘检测'''
    main(file_name)
    new_img = Image.open('canny.jpg')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)

    global img2
    img2.destroy()
    img2  = tkinter.Label(win,image=render)
    img2.image = render
    img2.place(x=800,y=100)


def load_image(image_path, load_size, crop_size):
    """加载单张图片并进行预处理"""
    transform = transforms.Compose([
        transforms.Resize(load_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加batch维度

def cycleganst():
    model_cycleGAN(file_name)
    image = Image.open('generate1.png')
    image = transforms.Resize((400,400))(image)
    tk_image = ImageTk.PhotoImage(image)

    # 显示生成的图片
    global img3
    img3 = tk.Label(win, image=tk_image)
    img3.image = tk_image  # 防止垃圾回收
    img3.place(x=800, y=100)


def generatorUNet():
    main(file_name)
    model()
    new_img = Image.open('generate.png')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)

    global img4
    # img4.destroy()
    img4  = tkinter.Label(win,image=render)
    img4.image = render
    img4.place(x=800,y=100)



e = tk.StringVar()
# e_entry = tk.Entry(win, width=68, textvariable=e)
# e_entry.pack()

button1 = tk.Button(win, text ="Select An Image", command = choose_file, font=("Arial", 13), 
                    fg="black", bg="blue", 
                    activebackground="green", 
                    activeforeground="yellow",
                    width=20,height =2)
button1.place(x=220,y=550)

button2 = tkinter.Button(win, text="edge detect" , command = edge_detect,width=10, height =2,
                        font=("Arial", 13), 
                        fg="black", bg="blue", 
                        activebackground="green", 
                        activeforeground="yellow",)
button2.place(x=570,y=200)

button3 = tkinter.Button(win, text="CycleGAN" , command = cycleganst,width=10, height =2,
                        font=("Arial", 13), 
                        fg="black", bg="blue", 
                        activebackground="green", 
                        activeforeground="yellow",)
button3.place(x=570,y=300)


button4 = tkinter.Button(win, text="GeneratorUNet" , command = generatorUNet, width=10, height =2,
                        font=("Arial", 13), 
                        fg="black", bg="blue", 
                        activebackground="green", 
                        activeforeground="yellow",)
button4.place(x=570,y=400)

label1 = tk.Label(win,text="Original Picture", font=("Arial", 14), bg='lightgray', fg='black')
label1.place(x=250,y=50)

label2 = tkinter.Label(win,text="Style Transfer!", font=("Arial", 14), bg='lightgray', fg='black')
label2.place(x=950,y=50)


# 退出按钮
button0 = tk.Button(win,text="Exit", command=win.quit, width=20, height =2, font=("Arial", 13),
                    fg="black", bg="blue", 
                    activebackground="green", 
                   activeforeground="yellow")
button0.place(x=570,y=650)
win.mainloop()
