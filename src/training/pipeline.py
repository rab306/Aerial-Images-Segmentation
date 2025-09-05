import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split

from src.config.settings import Config
from src.data.loaders import DataLoader
from src.data.preprocessors import ImagePreprocessor, MaskPreprocessor
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class AugmentationManager:
    """
    Handles data augmentation during training using Keras ImageDataGenerator.
    Keeps augmentation logic separate from data pipeline orchestration.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_gen_args = self._get_augmentation_config()
    
    def _get_augmentation_config(self) -> dict:
        """Get augmentation parameters from config."""
        return self.config.augmentation.to_dict()
    
    def create_training_generator(self, X_train: np.ndarray, y_train: np.ndarray, 
                                batch_size: int, seed: int = None):
        """
        Create synchronized training data generator with augmentation.
        
        Args:
            X_train: Training images
            y_train: Training masks
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            
        Returns:
            Generator that yields (image_batch, mask_batch)
        """
        if seed is None:
            seed = self.config.seed
            
        # Create generators for images and masks with same parameters
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        mask_datagen = ImageDataGenerator(**self.data_gen_args)
        
        # Create synchronized generators
        image_generator = image_datagen.flow(
            X_train, 
            batch_size=batch_size, 
            seed=seed,
            shuffle=True
        )
        
        mask_generator = mask_datagen.flow(
            y_train, 
            batch_size=batch_size, 
            seed=seed,
            shuffle=True
        )
        
        # Combined generator that yields both
        return self._combine_generators(image_generator, mask_generator)
    
    def create_validation_generator(self, X_val: np.ndarray, y_val: np.ndarray, 
                                  batch_size: int):
        """
        Create validation generator WITHOUT augmentation.
        
        Args:
            X_val: Validation images
            y_val: Validation masks
            batch_size: Batch size for validation
            
        Returns:
            Generator that yields (image_batch, mask_batch)
        """
        # No augmentation for validation - only basic preprocessing
        val_datagen = ImageDataGenerator()  # No augmentation parameters
        
        val_image_gen = val_datagen.flow(X_val, batch_size=batch_size, shuffle=False)
        val_mask_gen = val_datagen.flow(y_val, batch_size=batch_size, shuffle=False)
        
        return self._combine_generators(val_image_gen, val_mask_gen)
    
    def _combine_generators(self, image_gen, mask_gen):
        """
        Combine image and mask generators to ensure synchronization.
        
        Args:
            image_gen: Image generator
            mask_gen: Mask generator
            
        Yields:
            Tuple of (image_batch, mask_batch)
        """
        while True:
            try:
                image_batch = next(image_gen)
                mask_batch = next(mask_gen)
                yield image_batch, mask_batch
            except StopIteration:
                break


class TrainingDataPipeline:
    """
    Orchestrates the complete training data pipeline with proper data splitting
    to prevent data leakage. Coordinates data loading, preprocessing, and splitting.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize data components
        self.loader = DataLoader(config)
        self.img_processor = ImagePreprocessor(config)
        self.mask_processor = MaskPreprocessor(config)
        
        # Initialize augmentation manager
        self.augmentation_manager = AugmentationManager(config)
        
        # Split sample indices to prevent data leakage
        self.train_indices, self.val_indices, self.test_indices = self._split_sample_indices()
        
        print(f"Data split - Train: {len(self.train_indices)} samples, "
              f"Val: {len(self.val_indices)} samples, "
              f"Test: {len(self.test_indices)} samples")
    
    def _split_sample_indices(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Split sample indices to prevent data leakage.
        
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        total_samples = len(self.loader)
        all_indices = list(range(total_samples))
        
        # Use config values
        train_split = self.config.data_splits.train_split
        val_split = self.config.data_splits.val_split
        test_split = self.config.data_splits.test_split
        
        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            all_indices, 
            test_size=test_split,
            random_state=self.config.seed
        )
        
        # Second split: separate train and validation
        val_size = val_split / (train_split + val_split)  # Adjust for remaining data
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size,
            random_state=self.config.seed
        )
        
        return train_indices, val_indices, test_indices
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process training samples and return all patches.
        
        Returns:
            Tuple of (image_patches, mask_patches) for training
        """
        return self._process_split(self.train_indices, "Training")
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process validation samples and return all patches.
        
        Returns:
            Tuple of (image_patches, mask_patches) for validation
        """
        return self._process_split(self.val_indices, "Validation")
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process test samples and return all patches.
        
        Returns:
            Tuple of (image_patches, mask_patches) for testing
        """
        return self._process_split(self.test_indices, "Test")
    
    def get_training_generator(self, batch_size: int):
        """
        Get training data generator with augmentation.
        
        Args:
            batch_size: Training batch size
            
        Returns:
            Generator for training data
        """
        X_train, y_train = self.get_train_data()
        return self.augmentation_manager.create_training_generator(
            X_train, y_train, batch_size
        )
    
    def get_validation_generator(self, batch_size: int):
        """
        Get validation data generator without augmentation.
        
        Args:
            batch_size: Validation batch size
            
        Returns:
            Generator for validation data
        """
        X_val, y_val = self.get_val_data()
        return self.augmentation_manager.create_validation_generator(
            X_val, y_val, batch_size
        )
    
    def _process_split(self, indices: List[int], split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        image_patches = []
        mask_patches = []
        
        print(f"Processing {split_name.lower()} data: {len(indices)} samples...")
        
        for i, sample_idx in enumerate(indices):
            if i % 10 == 0:
                print(f"  {split_name} progress: {i}/{len(indices)} samples")
            
            # Load raw sample
            raw_img, raw_mask, metadata = self.loader.get_sample(sample_idx)
            
            # Process to patches
            img_patches = self.img_processor.process_image_to_patches(raw_img)
            sample_mask_patches = self.mask_processor.process_mask_to_patches(raw_mask)  # Rename to avoid confusion
            
            # Verify patch count matches
            if len(img_patches) != len(sample_mask_patches):
                raise ValueError(f"Patch count mismatch for sample {sample_idx}: "
                            f"{len(img_patches)} image patches vs {len(sample_mask_patches)} mask patches")
            
            # Collect patches
            image_patches.extend(img_patches)
            mask_patches.extend(sample_mask_patches)  # Fixed: extend with the new sample patches
        
        print(f"{split_name} processing complete: {len(image_patches)} total patches")
        
        return np.array(image_patches), np.array(mask_patches)
    
    def get_data_statistics(self) -> dict:
        """Get statistics about the data splits."""
        return {
            'total_samples': len(self.loader),
            'train_samples': len(self.train_indices),
            'val_samples': len(self.val_indices),
            'test_samples': len(self.test_indices),
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'test_indices': self.test_indices
        }

