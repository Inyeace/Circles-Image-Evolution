import os
import cv2
from matplotlib import pyplot as plt


def load_target_image(image_path, relative=True):
    """ Loads image from path, converts it to greyscale 

    Args:
        image_path(str): path to load the image.
        relative(boolean): whether path is relative to the input folder or absolute
    Returns:
        Image loaded from path as a numpy.ndarray.
    Raises:
        FileNotFoundError: image_path does not exist.
    """
    
    if relative:
        image_path = os.path.dirname(
            os.path.abspath(__file__)) + f'\input\{image_path}'

    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found at path: {image_path}')

    target = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return target 


def show_image(img_arr):
    """ Displays image on window
    Arguments:
        img_arr (numpy.ndarray): image array to be displayed
    """
    plt.figure()
    plt.axis("off")
    plt.imshow(img_arr, cmap="gray")
    plt.show()


