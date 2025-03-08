
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
import customtkinter as ctk


ctk.set_appearance_mode("Dark")
win = ctk.CTk()
win.title("Picture Process")
win.geometry("1280x1080")



original = Image.new('RGB', (300, 400))
save_img = Image.new('RGB', (300, 400))
count = 0
e2 = None
e2 = str(e2)
file_name = None
img2 = None
img3 = None
img4 = None
img5 = None

def choose_file():
    """
    Opens a file dialog to select an image file, then loads and displays it in a Tkinter window.

    Once the image file is selected, its path is stored in a global variable. The selected image
    is resized, converted to a Tkinter-compatible image, and then displayed in the window.

    Returns:
        None
    """
    select_file = tk.filedialog.askopenfilename(title='select the picture')
    global file_name
    file_name=select_file
    e.set(select_file)
    load = Image.open(select_file)
    load = transforms.Resize((400,400))(load)

    global original
    original = load
    render = ImageTk.PhotoImage(load)
    img  = tkinter.Label(win,image=render)
    img.image = render
    img.place(x=150,y=200)

def edge_detect():
    """
    Detect edges in the specified image by calling the main function, then load and
    display the resulting 'canny.jpg' in the application window at a resized
    dimension of 400x400. The old label is destroyed and replaced with a new label
    showing the updated image.
    Returns:
        None
    """
    
    main(file_name)
    new_img = Image.open('canny.jpg')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)
    global img2
    if img2:
        img2.destroy()
    img2  = tkinter.Label(win,image=render)
    img2.image = render
    img2.place(x=720,y=200)



def cycleganst(flowers=False, nonchar=False):
    """
    Generate and display a CycleGAN-transformed image in a Tkinter window.
    This function:
    1. Invokes the CycleGAN model with a specified file name to genreate the image.
    2. Loads the resulting image from 'generate1.png' and resizes it to 400x400 pixels.
    3. Converts the processed image into a Tkinter-compatible image object.
    4. Displays the transformed image within the Tkinter GUI.
    Returns:
        None
    """
    
    model_cycleGAN(file_name, flowers, nonchar)
    image = Image.open('generate1.png')
    image = transforms.Resize((400,400))(image)
    tk_image = ImageTk.PhotoImage(image)
    global img3
    if img3:
        img3.destroy()
    img3 = tk.Label(win, image=tk_image)
    img3.image = tk_image  
    img3.place(x=720, y=200)


def generatorUNet():
    main(file_name)
    model()
    new_img = Image.open('generate.png')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)
    global img4
    if img4:
        img4.destroy()
    img4  = tkinter.Label(win,image=render)
    img4.image = render
    img4.place(x=720,y=200)

def coloring():
    """
    Generate and display a CycleGAN-transformed image in a Tkinter window.
    This function:
    1. Invokes the CycleGAN model with a specified file name to genreate the image.
    2. Loads the resulting image from 'generate1.png' and resizes it to 400x400 pixels.
    3. Converts the processed image into a Tkinter-compatible image object.
    4. Displays the transformed image within the Tkinter GUI.
    Returns:
        None
    """
    model()
    new_img = Image.open('generate.png')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)
    global img5
    img5  = tkinter.Label(win,image=render)
    img5.image = render
    img5.place(x=720,y=200)

e = ctk.StringVar()


button1 = ctk.CTkButton(win,text="Select an Image", command=choose_file, font=("Arial", 15),
                        corner_radius=30, border_color="#288FB5", border_width=2,
                        fg_color="transparent", hover_color="#CCE7F0", text_color="#288FB5",
                        width=200, height=40)
button1.place(x=250,y=650)

button2 = ctk.CTkButton(win, text="Edge Detect", command=edge_detect, width=100, height=40,
                        font=("Arial", 15),
                        corner_radius=30, border_color="#3FB249", border_width=2,
                        fg_color="transparent", hover_color="#CCE7F0", text_color="#3FB249")
button2.place(x=480,y=650)

button3 = ctk.CTkButton(win, text="Photo", command=cycleganst, width=100, height=40,
                        font=("Arial", 15),
                        corner_radius=30, border_color="#C79BDA", border_width=2,
                        fg_color="transparent", hover_color="#CCE7F0", text_color="#C79BDA")
button3.place(x=620,y=650)

button4 = ctk.CTkButton(win, text="Flower Photo", command=lambda: cycleganst(flowers=True), width=100, height=40,
                        font=("Arial", 15),
                        corner_radius=30, border_color="#C79BDA", border_width=2,
                        fg_color="transparent", hover_color="#CCE7F0", text_color="#C79BDA")
button4.place(x=720,y=650)

button5 = ctk.CTkButton(win, text="Nonchar Photo", command=lambda: cycleganst(nonchar=True), width=100, height=40,
                        font=("Arial", 15),
                        corner_radius=30, border_color="#C79BDA", border_width=2,
                        fg_color="transparent", hover_color="#CCE7F0", text_color="#C79BDA")
button5.place(x=850,y=650)

button4 = ctk.CTkButton(win, text="Other Type", command=generatorUNet, width=100, height=40,
                        font=("Arial", 15),
                        corner_radius=30, border_color="#3FB249", border_width=2,
                        fg_color="transparent", hover_color="#CCE7F0", text_color="#3FB249")
button4.place(x=1005,y=650)



label1 = ctk.CTkLabel(win, text="Original Picture", font=("Arial", 24), text_color="white", pady=2, padx=2)
label1.place(x=255,y=150)

label2 = ctk.CTkLabel(win, text="Style Transfer!", font=("Arial", 24), text_color="white", pady=2, padx=2)
label2.place(x=850,y=150)

label3 = ctk.CTkLabel(win, text="Note: For the photo to painting, CycleGAN is used as transformation model", font=("Arial", 11), text_color="#F5F5F5", pady=3, padx=3)
label3.place(x=450,y=720)
label4 = ctk.CTkLabel(win, text="          For the edge dection, we use canny method.", font=("Arial", 11), text_color="#F5F5F5", pady=3, padx=3)
label4.place(x=450,y=740)
label5 = ctk.CTkLabel(win, text="          For the other type, Pix2Pix is used as transfermation model.", font=("Arial", 11), text_color="#F5F5F5", pady=3, padx=3)
label5.place(x=450,y=760)
# exit button
button0 = ctk.CTkButton(win, text="Exit", command=win.quit, width=150, height=40,
                        font=("Arial", 16),
                        corner_radius=30, 
                        fg_color="#288FB5", hover_color="#F84144", text_color="white")
button0.place(x=50,y=50)
win.mainloop()

