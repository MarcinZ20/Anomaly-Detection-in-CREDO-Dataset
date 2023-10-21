import numpy as np
import cv2
import skimage
import skimage.io as io
import time
from sklearn.linear_model import LinearRegression
import os

def otsu(image):                 #algorytm otsu automatycznie znajduje progowanie
    ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def dil_open(image_bin):         #dylatacja i opening automatycznie wygładza obszar znaleziony przez otsu

    image_bin = skimage.morphology.dilation(image_bin, np.ones((5,5)))
    image_bin = skimage.morphology.opening(image_bin, np.ones((5,5)))

    return image_bin

def masking(img, mask):         
    return cv2.bitwise_and(img, img, mask=mask)

def preprop_hachaj(img):       #preprocessing zaproponowany w pracy hachaja
    mask = otsu(img)
    mask = dil_open(mask)
    return masking(img, mask)

def mass_mean(image):          #przesuniecie obrazu żeby jego srodek kolorow był w centrum
    x_mean = 0
    y_mean = 0
    val = sum([sum(i) for i in image])
    for i in range(image.shape[0]):
        x_mean += sum(image[i, :])*i
    x_mean = x_mean/val
    for j in range(image.shape[1]):    
        y_mean += sum(image[:, j])*j
    y_mean = y_mean/val

    translation_matrix = np.array([[1, 0, 30 - y_mean], [0, 1, 30 - x_mean]], dtype=np.float32) 
    img = cv2.warpAffine(image, translation_matrix, (60, 60)) 
   
    return img

def main_line(img):       #znalezienie linii wzgledem której dokona się rotacji
    mask = otsu(img)

    data = np.nonzero(mask)

    reg = LinearRegression()
    reg.fit(data[0].reshape(-1, 1), data[1].reshape(-1, 1))
    return reg.coef_


def rotate(image):    #rotacja obrazu 
    a = main_line(image)
    (h, w) = image.shape[:2]
    cX, cY = (w // 2, h // 2)
    stopnie = int(np.arctan(-a)*180/np.pi)
    M = cv2.getRotationMatrix2D((cX, cY), stopnie, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def preprop(image):         #cały preprocessing zaproponowany przeze mnie
    image = preprop_hachaj(image)
    image = mass_mean(image)
    return rotate(image)


    






