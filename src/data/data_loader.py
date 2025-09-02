import numpy as np
import cv2
from patchify import patchify
from PIL import Image
from pathlib import Path
import logging

from Src.config import patch_size, directory

# Configure logging
logging.basicConfig(level=logging.ERROR)

def load_images():
    """
    Load and preprocess images from the specified directory.
    Returns a numpy array of normalized image patches.
    """
    image_dataset = []

    for path in Path(directory).glob('**/images'):
        images = path.glob('*.jpg')
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                logging.error(f"Error loading image: {path}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_to_patch(image)
            patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

            for row in range(patches_img.shape[0]):
                for col in range(patches_img.shape[1]):
                    single_patch_img = patches_img[row, col, :, :]
                    single_patch_img = normalize_image(single_patch_img)
                
                    
                    single_patch_img = single_patch_img[0]
                    image_dataset.append(single_patch_img)

    return np.array(image_dataset)


def load_masks():
    """
    Load and preprocess masks from the specified directory.
    Returns a numpy array of mask patches.
    """
    mask_dataset = []

    for path in Path(directory).glob('**/masks'):
        masks = sorted(path.glob('*.png')) 
        for mask_path in masks:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)

            if mask is None:
                logging.error(f"Error loading mask: {path}")
                continue

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = resize_to_patch(mask)
            patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)

            for row in range(patches_mask.shape[0]):
                for col in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[row, col, :, :]

                    single_patch_mask = single_patch_mask[0]
                    mask_dataset.append(single_patch_mask)

    return np.array(mask_dataset)


def resize_to_patch(image: np.ndarray) -> np.ndarray:
    """
    Resize the image to be divisible by the patch size.
    """
    x_size = (image.shape[0] // patch_size) * patch_size
    y_size = (image.shape[1] // patch_size) * patch_size
    image = Image.fromarray(image).crop((0, 0, x_size, y_size))
    return np.array(image)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image values between 0 and 1.
    """
    return image / 255.0



