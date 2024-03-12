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
    def __init__(self, img_dir, transform=None, ):

        self.img_dir = img_dir
        self.transform = transform
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

        if self.transform:
            image = norm(image) #Very important!!!
            image = image * 255
    
            image = masking(image, threshold_2(image))
            #image = masking(image, remove_dust)
            #img = mass_mean(img)
            #image = rotate(img)
            
        image = image.astype(np.float32)
        image = norm(image)
        
        return image, image
    

