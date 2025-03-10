import os

import cv2 as cv
import numpy as np
import glob
import sys
sys.setrecursionlimit(100000)  

def smooth(image, sigma = 1.4, length = 5):
    """ 
    Using gaussian filter to smooth the image
    
    Args:
        image: array of grey image
        sigma: the sigma of gaussian filter, default to be 1.4
        length: the kernal length, default to be 5

    Returns:
        the smoothed image
    """
    # create gaussian kernel
    k = length // 2
    gaussian = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            gaussian[i, j] = np.exp(-((i-k) ** 2 + (j-k) ** 2) / (2 * sigma ** 2))
    gaussian /= 2 * np.pi * sigma ** 2
    # Batch normalization
    gaussian = gaussian / np.sum(gaussian)

    # using convolution to smooth the image
    W, H = image.shape
    new_image = np.zeros([W - k * 2, H - k * 2])

    for i in range(W - 2 * k):
        for j in range(H - 2 * k):
            # convolution calculation
            new_image[i, j] = np.sum(image[i:i+length, j:j+length] * gaussian)

    new_image = np.uint8(new_image)
    return new_image

def get_gradient_and_direction(image):
    """ 
    Using sobel operator to calculate the gradients and directions
    Args:
        image: array of grey image

    Returns:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    W, H = image.shape
    gradients = np.zeros([W - 2, H - 2])
    direction = np.zeros([W - 2, H - 2])

    for i in range(W - 2):
        for j in range(H - 2):
            dx = np.sum(image[i:i+3, j:j+3] * Gx)
            dy = np.sum(image[i:i+3, j:j+3] * Gy)
            gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2)
            if dx == 0:
                direction[i, j] = np.pi / 2
            else:
                direction[i, j] = np.arctan(dy / dx)

    gradients = np.uint8(gradients)
    return gradients, direction

def NMS(gradients, direction):
    """ 
    Using Non-maximum suppression to suppress the gradients

    Args:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel

    Returns:
        the output image
    """
    W, H = gradients.shape
    nms = np.copy(gradients[1:-1, 1:-1])

    for i in range(1, W - 1):
        for j in range(1, H - 1):
            theta = direction[i, j]
            weight = np.tan(theta)
            if theta > np.pi / 4:
                d1 = [0, 1]
                d2 = [1, 1]
                weight = 1 / weight
            elif theta >= 0:
                d1 = [1, 0]
                d2 = [1, 1]
            elif theta >= - np.pi / 4:
                d1 = [1, 0]
                d2 = [1, -1]
                weight *= -1
            else:
                d1 = [0, -1]
                d2 = [1, -1]
                weight = -1 / weight

            g1 = gradients[i + d1[0], j + d1[1]]
            g2 = gradients[i + d2[0], j + d2[1]]
            g3 = gradients[i - d1[0], j - d1[1]]
            g4 = gradients[i - d2[0], j - d2[1]]

            grade_count1 = g1 * weight + g2 * (1 - weight)
            grade_count2 = g3 * weight + g4 * (1 - weight)

            if grade_count1 > gradients[i, j] or grade_count2 > gradients[i, j]:
                nms[i - 1, j - 1] = 0
    return nms

def double_threshold(nms, threshold1, threshold2):
    """ 
    Two thresholds calculation

    Args:
        nms: the input image
        threshold1: the low threshold
        threshold2: the high threshold

    Returns:
        The binary image.
    """
    visited = np.zeros_like(nms)
    output_image = nms.copy()
    W, H = output_image.shape

    def dfs(i, j):
        if i >= W or i < 0 or j >= H or j < 0 or visited[i, j] == 1:
            return
        visited[i, j] = 1
        if output_image[i, j] > threshold1:
            output_image[i, j] = 255
            dfs(i-1, j-1)
            dfs(i-1, j)
            dfs(i-1, j+1)
            dfs(i, j-1)
            dfs(i, j+1)
            dfs(i+1, j-1)
            dfs(i+1, j)
            dfs(i+1, j+1)
        else:
            output_image[i, j] = 0

    for w in range(W):
        for h in range(H):
            if visited[w, h] == 1:
                continue
            if output_image[w, h] >= threshold2:
                dfs(w, h)
            elif output_image[w, h] <= threshold1:
                output_image[w, h] = 0
                visited[w, h] = 1

    for w in range(W):
        for h in range(H):
            if visited[w, h] == 0:
                output_image[w, h] = 0
    return output_image

def enhancement(img):
    """
    enhance the image
    Args:
        img: the input image

    Returns:
        the output image
    """
    Imin, Imax = cv.minMaxLoc(img)[:2]
    # 使用numpy计算
    # Imax = np.max(img)
    # Imin = np.min(img)
    Omin, Omax = 0, 255
    # 计算a和b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * img + b
    out = out.astype(np.uint8)
    return out

def processing(img_path):

    image = cv.imread(img_path,0)
    image = enhancement(image)
    smoothed_image = smooth(image)
    gradients, direction = get_gradient_and_direction(smoothed_image)
    nms = NMS(gradients, direction)
    output_image = double_threshold(nms, 40, 100)

    return output_image

def main(input_path, output_path):
    files_path = []
    for label in glob.glob(f"{input_path}/*"):
        files_path.append(label)

    for file_path in files_path:
        out = os.path.join(output_path, file_path.split("/")[-1])
        print(file_path)
        image = processing(file_path)
        # img2 = image[:,:,::-1]
        # plt.imshow(image)
        print(out)
        cv.imwrite(out,image)
    

if __name__ == "__main__":
    
    input_path = "./test_image 2" 
    output_path = "./canny_pic"
    main(input_path, output_path)
