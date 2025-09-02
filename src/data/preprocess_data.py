import numpy as np

# Convert hexadecimal colors to RGB integer values
def hex_to_int(hex_color):
    """Converts a hexadecimal color string to an RGB integer array."""
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])

# Convert RGB mask to integer labels based on class colors
def rgb_to_2D_label(mask):
    """
    Convert RGB mask to integer labels based on predefined class colors.

    Args:
        mask (np.ndarray): Input mask in RGB format.

    Returns:
        np.ndarray: Segmentation mask with integer labels.
    """
    from Src.config import class_colors_hex
    class_colors = [hex_to_int(value) for value in class_colors_hex.values()]

    mask_seg = np.zeros(mask.shape[:-1], dtype=np.uint8)

    for idx, color in enumerate(class_colors):
        mask_seg[np.all(mask == color, axis=-1)] = idx 

    return mask_seg

def expand_mask_dims():
    """
    Expand the dimensions of the segmentation masks array.

    Returns:
        np.ndarray: Array of segmentation masks with an additional dimension.
    """
    from Src.data.data_loader import load_masks
    masks_array = load_masks()

    masks_seg_array = np.array([rgb_to_2D_label(mask) for mask in masks_array])
    masks_seg_array = np.expand_dims(masks_seg_array, axis=3)  
    return masks_seg_array

def one_hot_encoding():
    """
    Apply one-hot encoding to the segmentation mask labels.

    Returns:
        np.ndarray: One-hot encoded mask array.
    """
    from tensorflow.keras.utils import to_categorical
    from Src.config import num_classes
    mask_seg_array = expand_mask_dims()
    labels_categ = to_categorical(mask_seg_array, num_classes=num_classes) 
    return labels_categ

