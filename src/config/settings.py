from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


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


class PathsConfig:
    """Configuration for file and directory paths."""
    
    def __init__(self, output_dir: Optional[str] = None):
        # Use provided output directory or default
        base_dir = Path(output_dir) if output_dir else Path("output")
        
        # Directory structure
        self.models_dir = base_dir / "models"
        self.checkpoints_dir = base_dir / "checkpoints" 
        self.logs_dir = base_dir / "logs"
        self.results_dir = base_dir / "results"
        self.plots_dir = base_dir / "plots"
        
        # Store base directory and data directory for reference
        self.output_base = base_dir
        self.data_base = None  # Set later in Config.__init__
        
        # Create directories
        self.create_directories()
    
    def create_directories(self):
        """Create all configured directories."""
        directories = [
            self.models_dir, 
            self.checkpoints_dir, 
            self.logs_dir,
            self.results_dir, 
            self.plots_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"Created output directories under: {self.output_base}")
    
    def get_model_path(self, model_name: str = "unet", epoch: Optional[int] = None) -> Path:
        """Get full path for model file."""
        if epoch is not None:
            filename = f"{model_name}_epoch_{epoch:03d}.h5"
        else:
            filename = f"{model_name}_model.h5"
        return self.models_dir / filename
    
    def get_checkpoint_path(self, model_name: str = "unet") -> Path:
        """Get full path for checkpoint file."""
        # Use Keras naming convention with dot separator
        filename = f"{model_name.capitalize()}.weights.h5"
        return self.checkpoints_dir / filename
    
    def get_tensorboard_log_dir(self, model_name: str = "unet") -> Path:
        """Get TensorBoard log directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.logs_dir / f"{model_name}_{timestamp}"
    
    def get_training_plot_path(self, plot_name: str) -> Path:
        """Get path for training plots."""
        return self.plots_dir / f"{plot_name}.png"
    
    def get_evaluation_results_path(self, dataset_name: str = "test") -> Path:
        """Get path for evaluation results."""
        return self.results_dir / f"evaluation_{dataset_name}.json"


class Config:
    """Main configuration class for the aerial images segmentation project."""
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None, inference_mode: bool = False):
        # Data settings - use provided or default
        self.directory = data_dir or "raw_data/raw/"
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
        self.paths = PathsConfig(output_dir)
        
        # Set data base path for validation
        self.paths.data_base = Path(self.directory)
        
        # Store inference mode
        self.inference_mode = inference_mode
        
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
        
        # Validate data directory only if not in inference mode
        if not self.inference_mode:
            self._validate_data_directory()
    
    def _validate_data_directory(self):
        """Validate that data directory exists and has required structure."""
        data_path = Path(self.directory)
        
        if not data_path.exists():
            raise ValueError(f"Data directory does not exist: {data_path}")
        
        # Check for Tile directories
        tile_dirs = [d for d in data_path.iterdir() 
                    if d.is_dir() and d.name.startswith('Tile')]
        
        if not tile_dirs:
            raise ValueError(f"No 'Tile' directories found in: {data_path}")
        
        print(f"Found {len(tile_dirs)} Tile directories: {[d.name for d in sorted(tile_dirs)]}")
        
        # Validate that each Tile has images and masks directories
        for tile_dir in tile_dirs:
            images_dir = tile_dir / "images"
            masks_dir = tile_dir / "masks"
            
            if not images_dir.exists():
                raise ValueError(f"Missing 'images' directory in {tile_dir}")
            if not masks_dir.exists():
                raise ValueError(f"Missing 'masks' directory in {tile_dir}")
    
    def get_class_names(self):
        """Get list of class names in order."""
        return list(self.class_colors_hex.keys())
    
    def get_class_id(self, class_name: str) -> int:
        """Get the integer ID for a class name."""
        class_names = self.get_class_names()
        if class_name not in class_names:
            raise ValueError(f"Unknown class name: {class_name}")
        return class_names.index(class_name)
    
    def save_config(self, filepath: Optional[str] = None) -> Path:
        """Save configuration to JSON file."""
        if filepath is None:
            filepath = self.paths.output_base / "config.json"
        else:
            filepath = Path(filepath)
        
        config_dict = {
            'data_directory': str(self.directory),
            'patch_size': self.patch_size,
            'num_channels': self.num_channels,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'class_names': self.get_class_names(),
            'class_colors': self.class_colors_hex,
            'data_splits': {
                'train': self.data_splits.train_split,
                'val': self.data_splits.val_split,
                'test': self.data_splits.test_split
            },
            'augmentation': self.augmentation.to_dict(),
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to: {filepath}")
        return filepath
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create new config with basic parameters
        config = cls(
            data_dir=config_dict['data_directory'],
            output_dir=None,  # Will use default
            inference_mode=True  # Skip validation when loading
        )
        
        # Override with saved values
        config.patch_size = config_dict['patch_size']
        config.batch_size = config_dict['batch_size']
        config.epochs = config_dict['epochs']
        
        return config
    
    def print_summary(self):
        """Print configuration summary."""
        print("=== Configuration Summary ===")
        print(f"Data directory: {self.directory}")
        print(f"Output directory: {self.paths.output_base}")
        print(f"Patch size: {self.patch_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.get_class_names()}")
        print(f"Data splits: Train={self.data_splits.train_split}, "
              f"Val={self.data_splits.val_split}, Test={self.data_splits.test_split}")
        print(f"Augmentation enabled: {self.augmentation.rotation_range > 0}")
        print(f"Models will be saved to: {self.paths.models_dir}")
        print(f"Checkpoints will be saved to: {self.paths.checkpoints_dir}")