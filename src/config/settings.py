from pathlib import Path
from typing import Dict


class AugmentationConfig:
    """Configuration for data augmentation parameters."""
    
    def __init__(self):
        self.rotation_range = 45.0
        self.width_shift_range = 0.2
        self.height_shift_range = 0.2
        self.zoom_range = 0.2
        self.horizontal_flip = True
        self.vertical_flip = True
        self.fill_mode = 'nearest'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for Keras ImageDataGenerator."""
        return {
            'rotation_range': self.rotation_range,
            'width_shift_range': self.width_shift_range,
            'height_shift_range': self.height_shift_range,
            'zoom_range': self.zoom_range,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'fill_mode': self.fill_mode
        }


class DataSplitConfig:
    """Configuration for train/validation/test data splitting."""
    
    def __init__(self):
        self.train_split = 0.7
        self.val_split = 0.2
        self.test_split = 0.1
    
    def validate_splits(self):
        """Ensure splits sum to 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Data splits must sum to 1.0, got {total}")


class Config:
    """Main configuration class for the satellite segmentation project."""
    
    def __init__(self):
        # Data settings
        self.directory = "raw_data/raw/"
        self.patch_size = 256
        self.num_channels = 3
        self.num_classes = 6
        
        # Training settings
        self.batch_size = 32
        self.epochs = 500
        self.verbose = 1
        self.seed = 42
        self.smooth = 1e-8
        
        # Loss function parameters 
        self.focal_alpha = 0.5
        self.focal_gamma = 1.0
        
        # File paths
        self.checkpoint_path = 'Unet.weights.h5'
        self.model_path = 'Unet_model.h5'

        # Metrics settings
        self.primary_monitor_metric = 'val_jaccard_coeff'
        self.secondary_monitor_metric = 'val_mean_iou'  
        self.patience = 100
        
        # Class information
        self.class_colors_hex = {
            'Building': '#3C1098',
            'Land': '#8429F6',
            'Road': '#6EC1E4',
            'Vegetation': '#FEDD3A',
            'Water': '#E2A929',
            'Unlabeled': '#9B9B9B'
        }
        
        # Nested configurations
        self.augmentation = AugmentationConfig()
        self.data_splits = DataSplitConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.patch_size <= 0:
            raise ValueError("Patch size must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.num_classes != len(self.class_colors_hex):
            raise ValueError(f"Number of classes ({self.num_classes}) must match "
                           f"number of class colors ({len(self.class_colors_hex)})")
        
        # Validate data splits
        self.data_splits.validate_splits()
        
        # Validate directory exists
        data_path = Path(self.directory)
        if not data_path.exists():
            print(f"Warning: Data directory does not exist: {data_path}")
    
    def get_class_names(self):
        """Get list of class names in order."""
        return list(self.class_colors_hex.keys())
    
    def get_class_id(self, class_name: str) -> int:
        """Get the integer ID for a class name."""
        class_names = self.get_class_names()
        if class_name not in class_names:
            raise ValueError(f"Unknown class name: {class_name}")
        return class_names.index(class_name)
    
    def print_summary(self):
        """Print configuration summary."""
        print("=== Configuration Summary ===")
        print(f"Data directory: {self.directory}")
        print(f"Patch size: {self.patch_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.get_class_names()}")
        print(f"Data splits: Train={self.data_splits.train_split}, "
              f"Val={self.data_splits.val_split}, Test={self.data_splits.test_split}")
        print(f"Augmentation enabled: {self.augmentation.rotation_range > 0}")