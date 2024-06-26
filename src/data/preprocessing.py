import numpy as np
import cv2
import skimage
from sklearn.linear_model import LinearRegression
from skimage.segmentation import flood
from skimage.morphology import opening
from sklearn.decomposition import PCA
import math
from math import atan2

def norm(image):
    image = (image - image.min())/(image.max()- image.min())
    return image

def otsu(image: np.ndarray) -> np.ndarray:
    """OTSU algorithm automatically finds the thresholding

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: optimal threshold value
    """
    image = image.astype(np.uint8)

    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def threshold_2(image: np.ndarray) -> np.ndarray:
    
    image = image.astype(np.uint8)
    return cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)[1]

def threshold_3(image: np.ndarray) -> np.ndarray:
    
    image = image.astype(np.uint8)
    return cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)[1]

def remove_dust(image: np.ndarray) -> np.ndarray:
    """ tup = (29, 29)
    img = image
    if image[29, 29] > 250:
        tup = (29, 29)
    elif image[29, 30]>250:
        tup = (29, 30)
    elif image[30, 29]>250:
        tup = (30, 29)
    elif image[30, 30]>250:
        tup = (30, 30) """
    
    img = image
    center = image[27:33, 27:33]
    
    if len(np.where(center>250))>1:
        #print(np.where(center > 250))
        y, x = np.where(center > 250)
        if y.size == 0:
            return image
        seed = (27+y[0], 27+x[0])
        mask = flood(image, seed, tolerance = 240)
        
        maska = np.invert(mask)
        maska = maska*1
        maska = maska.astype('uint8')
        temp = masking(image, maska)
        temp = opening(temp)
        image = np.logical_or(temp, masking(image, mask.astype('uint8')))
        image = masking(img, image.astype('uint8'))
    return image


def dil_open(image_bin: np.ndarray) -> np.ndarray:
    """Dylatacja i opening automatycznie wygładza obszar znaleziony przez OTSU

    Args:
        image_bin (np.ndarray): input image

    Returns:
        np.ndarray: output image
    """
    image_bin = skimage.morphology.dilation(image_bin, np.ones((2, 2)))  #wcześniej było (5, 5)
    image_bin = skimage.morphology.opening(image_bin, np.ones((2, 2)))

    return image_bin


def masking(img: np.ndarray, mask: np.ndarray):
    return cv2.bitwise_and(img, img, mask=mask)


def preprop_hachaj(img: np.ndarray):
    mask = otsu(img)
    mask = dil_open(mask)
    return masking(img, mask)


def mass_mean(image: np.ndarray, borderMode=cv2.BORDER_CONSTANT) -> np.ndarray:
    x_mean = 0
    y_mean = 0
    val = sum([sum(i) for i in image])

    if val == 0:
        return image

    for i in range(image.shape[0]):
        x_mean += sum(image[i, :]) * i

    x_mean = x_mean / val

    for j in range(image.shape[1]):
        y_mean += sum(image[:, j]) * j

    y_mean = y_mean / val

    translation_matrix = np.array(
        [[1, 0, 30 - y_mean], [0, 1, 30 - x_mean]], dtype=np.float32
    )

    return cv2.warpAffine(image, translation_matrix, (60, 60), borderMode = borderMode)


def main_line(img: np.ndarray) -> np.float32:
    mask = otsu(img)

    data = np.nonzero(mask)
    reg = LinearRegression()

    if len(data[0]) == 0:
        return 0

    reg.fit(data[0].reshape(-1, 1), data[1].reshape(-1, 1))

    return reg.coef_

def main_line_weights(img: np.ndarray) -> np.float32:

    img_copy = np.copy(img).flatten()
    data = np.nonzero(img)
    weigths = img_copy[img_copy!=0]

    reg = LinearRegression()

    if len(data[0]) == 0:
        return 0

    reg.fit(data[0].reshape(-1, 1), data[1].reshape(-1, 1), sample_weight=weigths)

    return reg.coef_



def rotate(image: np.ndarray, weigthed = False, borderMode=cv2.BORDER_CONSTANT) -> np.ndarray: #cv2.BORDER_REFLECT101 #cv2.BORDER_REPLICATE #cv2.BORDER_CONSTANT
    if weigthed:
        a = main_line_weights(image)
    else:
        a = main_line(image)
    (h, w) = image.shape[:2]
    cX, cY = (w // 2, h // 2)
    stopnie = int(np.arctan(-a) * 180 / np.pi)
    M = cv2.getRotationMatrix2D((cX, cY), stopnie, 1.0)

    return cv2.warpAffine(image, M, (w, h), borderMode = borderMode)

def better_binarize(image):

    image = image.astype(np.uint8)
    hist_data, bins = np.histogram(image.flatten(), bins = 51)
    max_index = np.where(hist_data == hist_data.max())[0][-1]
    thresh = int((255 - bins[max_index+1])/2 + bins[max_index+1])
    return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

def preprop_2(image: np.ndarray) -> np.ndarray:
    """
    First preprocessing presented by us

    Args: image: np.ndarray

    Returns: np.ndarray: Processed image
    """
    image = preprop_hachaj(image)
    image = mass_mean(image)

    return rotate(image)

def preprop(image: np.ndarray) -> np.ndarray:
    """
    Current preprocessing presented by us

    Args: image: np.ndarray

    Returns: np.ndarray: Processed image
    """
    image = norm(image) #Very important!!!
    image = image * 255
    image = masking(image, threshold_2(image))
    image = remove_dust(image)
    image = mass_mean(image)
    return rotate(image)

def better_preprop(image):

    image = norm(image) #Very important!!!
    image = image * 255
    image = masking(image, better_binarize(image))
    image = remove_dust(image)
    image = mass_mean(image)
    return rotate(image)


def simple_preprop(image: np.ndarray) -> np.ndarray:

    image = norm(image) #Very important!!!
    image = mass_mean(image)
    return rotate(image)

def prepro_hachaj_for_real(image):

    #image = norm(image)
    image = mass_mean(image)
    return rotate(image, weigthed=True)

def align_image_2(img, borderMode = cv2.BORDER_CONSTANT):
    
    gray = np.copy(img)
    my_list = []
    for x in range(gray.shape[0]):
        for y in range(gray.shape[1]):
            z = 0
            while z < gray[x,y]:
                my_list.append([y, x])
                z = z + 1

    X = np.array(my_list)
    pca = PCA(n_components=2)
    pca.fit(X)
    mean = pca.mean_
    eigenvectors = pca.components_

    cntr = (int(mean[0]), int(mean[1]))
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    angle = 180 * angle / math.pi

    (cX, cY) = cntr
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    (h, w) = gray.shape[:2]

    rotated = cv2.warpAffine(gray, M, (w, h), borderMode = borderMode)

    xx = w / 2 - cX
    yy = h / 2 - cY
    M = np.float32([[1, 0, xx], [0, 1, yy]])

    shifted = cv2.warpAffine(rotated, M, (rotated.shape[1], rotated.shape[0]), borderMode = borderMode)

    return shifted
