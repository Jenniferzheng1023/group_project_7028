
import tkinter as tk
import tkinter.filedialog
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from PIL import Image, ImageTk
from torchvision import transforms as transforms
from test import main, model
from data import create_dataset
from models import create_model
import cv2 as cv
import os
import torch

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
    
    image_tensor = load_image(file_name, 512, 512)
    print(image_tensor)
    print(f"File path: {file_name}")
    load = Image.open(file_name)
    print(f"Loaded image: {load}")

    model = create_model(gpu_ids='', isTrain=False, name='latest_net_G', model='test')
    # 加载预训练的权重（使用 BaseModel 的 load_networks 方法）
    model.load_networks('latest')  # 假设权重文件名为 latest_net_*.pth

    # 设置模型，确保运行在 CPU 上
    model.setup()

    # 移动生成器到设备（CPU）
    if hasattr(model, 'netG'):  # 检查是否有 netG_A 属性
        model.netG.to(model.device)
    else:
        raise AttributeError("The model does not have a 'netG' attribute.")
    
    # 设置输入并进行推理
    model.set_input({'A': image_tensor, 'A_paths': file_name})
    model.test()

    visuals = model.get_current_visuals()
    fake_image = visuals['fake']
    tensor = fake_image.squeeze(0)

    # 2. 确保值在 [0, 1] 范围内
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    # 3. 转换为 PIL 图像
    image = ToPILImage()(tensor)
    # 将 PIL 图像转换为 Tkinter 支持的 PhotoImage 对象
    tk_image = ImageTk.PhotoImage(image)

    # 显示生成的图片
    global img3
    img3 = tk.Label(win, image=tk_image)
    img3.image = tk_image  # 防止垃圾回收
    img3.place(x=700, y=100)

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

button2 = tkinter.Button(win, text="CycleGAN" , command = cycleganst,width=10, height =2,
                        font=("Arial", 13), 
                        fg="black", bg="blue", 
                        activebackground="green", 
                        activeforeground="yellow",)
button2.place(x=570,y=100)

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
