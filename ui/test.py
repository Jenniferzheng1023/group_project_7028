import argparse

import cv2
import numpy
import PIL.Image
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from models import *
from canny import processing
from models_generatorUNet import *
from picture2texture import estimate
import numpy as np
from skimage import morphology

def sample_images(generator,Tensor,imgs):
    """
    save the processed pictures

    Args:
        generator: trained model
        Tensor: tensor format
        imgs: real picture
    Returns:
        None
    """
    real_A = Variable(imgs.type(Tensor))
    real_A = real_A.unsqueeze(0)
    fake_B = generator(real_A)
    # cv2.imwrite("generate.png" ,255*fake_B[0].squeeze(0).cpu().swapaxes(0,2).swapaxes(0,1).numpy())
    cv2.imwrite("generate.png", 255 * fake_B[0].squeeze(0).cpu().detach().swapaxes(0, 2).swapaxes(0, 1).numpy())
    
def process(opt,file_path):
    """
    get the HED edge-painting

    Args:
        opt: opt file
        file_path: the file path U want to process
    
    Returns:
        None
    """
    arguments_strOut = "HED.jpg"
    src = cv2.imread(file_path, 0)
    src = cv2.resize(src, (opt.img_width,opt.img_height))
    src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    a = PIL.Image.fromarray(src_RGB)
    b = numpy.array(a)[:, :]
    tenInput = torch.FloatTensor(numpy.ascontiguousarray(b.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenOutput = estimate(tenInput)
    PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)

def main(path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    opt = parser.parse_args()

    process(opt,path) # process the picture to HED edge-painting
    # post_process() # post-process the picture
    img = processing(path) # process the picture to canny edge-painting
    cv2.imwrite("canny.jpg",img)
    pic1 = cv2.imread("HED.jpg")
    pic1 = cv2.resize(pic1, (opt.img_width,opt.img_height))
    # pic1 = cv2.imread("generate_post.jpg")
    # pic1 = cv2.resize(pic1, (opt.img_width,opt.img_height))
    pic2 = cv2.imread("canny.jpg")
    pic2 = cv2.resize(pic2, (opt.img_width,opt.img_height))
    train_data = pic2+pic1
    cv2.imwrite("canny&HED.jpg",train_data) # gain the picture which is the combination of canny and HED

def load_image(image_path, load_size, crop_size):
    """
    load the image to preprocess
    Args:
        image_path (_type_): original image path
        load_size (_type_): the size of the image
        crop_size (_type_): the size of the crop

    Returns:
        _type_: the processed image
    """
    transform = transforms.Compose([
        transforms.Resize(load_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # add batch dimension

def model_cycleGAN(file_name, flowers, nonchar):
    image_tensor = load_image(file_name, 512, 512)
    # print(f"File path: {file_name}")
    load = Image.open(file_name)
    # print(f"Loaded image: {load}")
    
    if flowers:
        model = create_model(gpu_ids='', isTrain=False, name='latest_net_G_flowers', model='test')
    elif nonchar:
        model = create_model(gpu_ids='', isTrain=False, name='latest_net_G_nonchar', model='test')
    else:
        model = create_model(gpu_ids='', isTrain=False, name='latest_net_G', model='test')
    
    # model = create_model(gpu_ids='', isTrain=False, name='latest_net_G', model='test')

    # loading the pre-trained weights (using the load_networks method of BaseModel)
    model.load_networks('latest')  

    # step1: set the model
    model.setup()
    if hasattr(model, 'netG'):  # check whether the model has netG
        model.netG.to(model.device)
    else:
        raise AttributeError("The model does not have a 'netG' attribute.")
    
    model.set_input({'A': image_tensor, 'A_paths': file_name})
    model.test()

    visuals = model.get_current_visuals()
    fake_image = visuals['fake']
    tensor = fake_image.squeeze(0)

    # 2. make sure the tensor is in the range
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1) 
    
    # 3. convert the tensor to an image and save it
    to_pil = ToPILImage()
    image = to_pil(tensor.squeeze(0).cpu()) 
    image.save('generate1.png')

# def post_process():
#     """
#     post-process the picture
#     """
#     img = cv2.imread('HED.jpg', 0)
#     img = np.asarray(img)
#     binarized = cv2.adaptiveThreshold(img, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
#     binarized = np.asarray(binarized, np.float32) / 255
#     thinned = morphology.thin(binarized)
#     cleaned = morphology.remove_small_objects(thinned, min_size=128, connectivity=2)
#     cleaned_uint8 = (cleaned * 255).astype(np.uint8)
#     cv2.imwrite("generate_post.jpg", cleaned_uint8)



def model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    opt = parser.parse_args()

    transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])

    cuda = True if torch.cuda.is_available() else False
    generator = GeneratorUNet()
    if cuda:
        generator = generator.cuda() 
    # ! model path is there
    generator.load_state_dict(torch.load("./checkpoints/generator_45_canny.pth", map_location='cpu'))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    train_data = cv2.imread("canny&HED.jpg")
    frame = cv2.resize(train_data,(opt.img_width,opt.img_height))
    frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    frame = transform(frame)
    sample_images(generator,Tensor,frame) # to gain the processed picture

if __name__ == "__main__":
    path = "1.jpg" 
    main(path)
