import os
import cv2
import numpy as np

def load_target_image(image_path: str,size = None) -> np.ndarray:
    """ Loads image from path, converts it to greyscale 

    Args:
        image_path(str): path to load the image.
        relative(boolean): whether path is relative to the input folder or absolute
    Returns:
        Image loaded from path as a numpy.ndarray.
    Raises:
        FileNotFoundError: image_path does not exist.
    """
    
    
    image_path = os.path.dirname(
        os.path.abspath(__file__)) + f'\input\{image_path}'

    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found at path: {image_path}')

    target = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if size:
        target = cv2.resize(src=target, dsize=size,interpolation=cv2.INTER_AREA)
    return target 

def save_image(img_name: str,img_arr: np.ndarray) -> bool:
    image_path = os.path.dirname(
        os.path.abspath(__file__)) + f'\output\{img_name}.jpg'
    return cv2.imwrite(image_path,img_arr)




