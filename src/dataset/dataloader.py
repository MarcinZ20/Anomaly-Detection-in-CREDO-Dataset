from src.data.preprocessing import *
import torch 
import cv2
import numpy as np
import os
from skimage import io

def norm(image):
    image = (image - image.min())/(image.max()- image.min())
    return image


class CREDO_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None,align =False,conv = False):
        self.conv = conv
        self.img_dir = img_dir
        self.transform = transform
        self.align = align
        self.dics = os.listdir(self.img_dir)
        self.dics.pop(0)
        self.files = []
        for i in self.dics:
            for img in os.listdir(os.path.join(self.img_dir, i)):
                self.files.append(os.path.join(i, img))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = io.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if self.align:
            
            image = align_image_2(image)

        elif self.transform:

            image = preprop(image)
        
        if self.conv:

            temp = np.zeros((64, 64))
            temp[2:62, 2:62] = image
            image = temp
  
        image = image.astype(np.float32)
        image = norm(image)
        
        return image, self.files[idx]
    
class CREDO_Small_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, conv = False):

        self.conv = conv
        self.img_dir = img_dir
        self.transform = transform
        self.files = [file for file in os.listdir(img_dir)]
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = io.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.transform:
            #image = norm(image) #Very important!!!
            #image = image * 255
            #image = masking(image, threshold_2(image))
            #
            image = preprop(image)
        
        if self.conv:

            temp = np.zeros((64, 64))
            temp[2:62, 2:62] = image
            image = temp
        
        image = image.astype(np.float32)
        image = norm(image)
        
        return image, image
    

