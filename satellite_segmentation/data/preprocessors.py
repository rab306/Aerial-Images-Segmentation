import numpy as np
from PIL import Image
from patchify import patchify
from typing import List, Tuple
from tensorflow.keras.utils import to_categorical

# Utility functions (implementation)
def resize_to_patch_size(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Utility function for resizing."""
    height, width = image.shape[:2]
    new_height = (height // patch_size) * patch_size
    new_width = (width // patch_size) * patch_size
    
    if new_height == 0 or new_width == 0:
        raise ValueError(f"Image too small for patch size {patch_size}")
    
    return image[:new_height, :new_width]

def create_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Utility function for creating patches."""
    return patchify(image, (patch_size, patch_size, 3), step=patch_size)

# Class that uses utility functions
class ImagePreprocessor:
    def __init__(self, config):
        self.patch_size = config.patch_size
    
    def resize_to_patch_size(self, image: np.ndarray) -> np.ndarray:
        """Delegate to utility function."""
        return resize_to_patch_size(image, self.patch_size)
    
    def create_patches(self, image: np.ndarray) -> np.ndarray:
        """Delegate to utility function."""
        return create_patches(image, self.patch_size)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Image-specific logic - no utility needed."""
        return image.astype(np.float32) / 255.0
    
    def process_image_to_patches(self, raw_image: np.ndarray) -> List[np.ndarray]:
        """Complete pipeline using class methods."""
        resized = self.resize_to_patch_size(raw_image)  # Uses utility via delegation
        patches = self.create_patches(resized)          # Uses utility via delegation
        
        processed_patches = []
        for row in range(patches.shape[0]):
            for col in range(patches.shape[1]):
                patch = patches[row, col, 0]
                normalized_patch = self.normalize(patch)  # Image-specific method
                processed_patches.append(normalized_patch)
        
        return processed_patches
    

class MaskPreprocessor:
    def __init__(self, config):
        self.config = config
        self.class_colors_hex = config.class_colors_hex
        self.num_classes = config.num_classes
        self.patch_size = config.patch_size
        
        # Pre-compute RGB values for efficiency
        self.class_colors_rgb = self._hex_to_rgb_mapping()
    
    def _hex_to_rgb_mapping(self):
        """Convert hex colors to RGB values."""
        rgb_colors = []
        for hex_color in self.class_colors_hex.values():
            hex_color = hex_color.lstrip('#')
            rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
            rgb_colors.append(rgb)
        return rgb_colors
    
    def rgb_to_labels(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert RGB mask to integer labels.
        
        Args:
            mask: RGB mask array (H, W, 3)
        Returns:
            Label mask array (H, W) with class indices
        """
        label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        
        for class_idx, rgb_color in enumerate(self.class_colors_rgb):
            # Find pixels that match this class color
            matches = np.all(mask == rgb_color, axis=-1)
            label_mask[matches] = class_idx
            
        return label_mask
    
    def to_one_hot(self, label_mask: np.ndarray) -> np.ndarray:
        """
        Convert label mask to one-hot encoding.
        
        Args:
            label_mask: Label array (H, W) 
        Returns:
            One-hot encoded array (H, W, num_classes)
        """
        # Add channel dimension for to_categorical
        label_mask_expanded = np.expand_dims(label_mask, axis=-1)
        one_hot = to_categorical(label_mask_expanded, num_classes=self.num_classes)
        return one_hot
    
    def resize_to_patch_size(self, image: np.ndarray) -> np.ndarray:
        """Delegate to utility function."""
        return resize_to_patch_size(image, self.patch_size)
    
    def create_patches(self, image: np.ndarray) -> np.ndarray:
        """Delegate to utility function."""
        return create_patches(image, self.patch_size)
    
    def process_mask_to_patches(self, raw_mask: np.ndarray) -> List[np.ndarray]:
        """Complete mask processing: resize -> patch -> rgb_to_labels -> one_hot."""
        resized = self.resize_to_patch_size(raw_mask)
        patches = self.create_patches(resized)
        
        processed_patches = []
        for row in range(patches.shape[0]):
            for col in range(patches.shape[1]):
                patch = patches[row, col, 0]
                
                # Actually process the mask
                label_patch = self.rgb_to_labels(patch)
                one_hot_patch = self.to_one_hot(label_patch)
                
                processed_patches.append(one_hot_patch)
        
        return processed_patches
    