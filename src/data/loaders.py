from satellite_segmentation.config.settings import Config
from pathlib import Path
import cv2
import json
import numpy as np
from typing import List, Tuple, Optional


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_root = Path(config.directory)
        
        # Discover all tiles and their data
        self.paired_data = self._discover_and_pair_data()
        print(f"Found {len(self.paired_data)} image/mask pairs across all tiles")
    
    def _discover_and_pair_data(self) -> List[Tuple[Path, Path, str]]:
        """
        Discover and pair image/mask files across all tiles.
        Returns: List of (image_path, mask_path, tile_name) tuples
        """
        paired_data = []
        
        # Find all tile directories
        tile_dirs = [d for d in self.data_root.iterdir() 
                    if d.is_dir() and d.name.startswith('Tile')]
        
        for tile_dir in sorted(tile_dirs):
            images_dir = tile_dir / "images"
            masks_dir = tile_dir / "masks"
            
            if not (images_dir.exists() and masks_dir.exists()):
                print(f"Warning: Missing images or masks directory in {tile_dir.name}")
                continue
            
            # Find all images in this tile
            image_files = list(images_dir.glob("image_part_*.jpg"))
            
            for img_path in sorted(image_files):
                # Find corresponding mask
                mask_name = img_path.stem + ".png"  
                mask_path = masks_dir / mask_name
                
                if mask_path.exists():
                    paired_data.append((img_path, mask_path, tile_dir.name))
                else:
                    print(f"Warning: Missing mask for {img_path.name} in {tile_dir.name}")
        
        return paired_data
    
    def __len__(self) -> int:
        """Return total number of samples across all tiles."""
        return len(self.paired_data)
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load single image/mask pair on demand.
        
        Returns:
            image: RGB image array
            mask: Mask array  
            metadata: Dict with tile info, file names, etc.
        """
        if index >= len(self.paired_data):
            raise IndexError(f"Index {index} out of range for dataset size {len(self)}")
        
        img_path, mask_path, tile_name = self.paired_data[index]
        
        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Metadata
        metadata = {
            'tile_name': tile_name,
            'image_name': img_path.name,
            'mask_name': mask_path.name,
            'image_path': str(img_path),
            'mask_path': str(mask_path)
        }
        
        return image, mask, metadata
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Load a batch of samples."""
        batch_images = []
        batch_masks = []
        batch_metadata = []
        
        for idx in indices:
            image, mask, metadata = self.get_sample(idx)
            batch_images.append(image)
            batch_masks.append(mask)
            batch_metadata.append(metadata)
        
        return np.array(batch_images), np.array(batch_masks), batch_metadata
    