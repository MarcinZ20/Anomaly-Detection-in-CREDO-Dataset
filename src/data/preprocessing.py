import numpy as np
import cv2
import skimage
from sklearn.linear_model import LinearRegression


def otsu(image: np.ndarray) -> np.ndarray:
    """OTSU algorithm automatically finds the thresholding

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: optimal threshold value
    """
    image = image.astype(np.uint8)

    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def dil_open(image_bin: np.ndarray) -> np.ndarray:
    """Dylatacja i opening automatycznie wygładza obszar znaleziony przez OTSU

    Args:
        image_bin (np.ndarray): input image

    Returns:
        np.ndarray: output image
    """
    image_bin = skimage.morphology.dilation(image_bin, np.ones((5, 5)))
    image_bin = skimage.morphology.opening(image_bin, np.ones((5, 5)))

    return image_bin


def masking(img: np.ndarray, mask: np.ndarray):
    return cv2.bitwise_and(img, img, mask=mask)


def preprop_hachaj(img: np.ndarray):
    mask = otsu(img)
    mask = dil_open(mask)
    return masking(img, mask)


def mass_mean(image: np.ndarray) -> np.ndarray:
    x_mean = 0
    y_mean = 0
    val = sum([sum(i) for i in image])

    for i in range(image.shape[0]):
        x_mean += sum(image[i, :]) * i

    x_mean = x_mean / val

    for j in range(image.shape[1]):
        y_mean += sum(image[:, j]) * j

    y_mean = y_mean / val

    translation_matrix = np.array(
        [[1, 0, 30 - y_mean], [0, 1, 30 - x_mean]], dtype=np.float32
    )

    return cv2.warpAffine(image, translation_matrix, (60, 60))


def main_line(img: np.ndarray) -> np.float32:
    mask = otsu(img)

    data = np.nonzero(mask)

    reg = LinearRegression()
    reg.fit(data[0].reshape(-1, 1), data[1].reshape(-1, 1))

    return reg.coef_


def rotate(image: np.ndarray) -> np.ndarray:
    a = main_line(image)
    (h, w) = image.shape[:2]
    cX, cY = (w // 2, h // 2)
    stopnie = int(np.arctan(-a) * 180 / np.pi)
    M = cv2.getRotationMatrix2D((cX, cY), stopnie, 1.0)

    return cv2.warpAffine(image, M, (w, h))


def preprop(image: np.ndarray) -> np.ndarray:
    """
    Whole preprocessing presented by us

    Args: image: np.ndarray

    Returns: np.ndarray: Processed image
    """
    image = preprop_hachaj(image)
    image = mass_mean(image)

    return rotate(image)
