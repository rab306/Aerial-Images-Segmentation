import time
from typing import Tuple, Optional, Dict, Any
from keras.models import Model
from keras.callbacks import History

from src.config.settings import Config
from src.models.factory import ModelFactory
from src.training.pipeline import TrainingDataPipeline
from src.training.components import CallbacksManager


class ModelTrainer:
    """
    Orchestrates the complete model training process with dependency injection.
    
    Handles training workflow while delegating specific responsibilities to
    injected components for better testability and flexibility.
    """
    
    def __init__(self, config: Config):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration object containing all training parameters
        """
        self.config = config
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.data_pipeline = TrainingDataPipeline(config)
        self.callbacks_manager = CallbacksManager(config)
        
        # Training state
        self.model: Optional[Model] = None
        self.training_history: Optional[History] = None
        
    def train(self, model_type: str = "unet", save_model: bool = True) -> Tuple[Model, History]:
        """
        Execute the complete training process.
        
        Args:
            model_type: Type of model to train (default: "unet")
            save_model: Whether to save the trained model (default: True)
            
        Returns:
            Tuple of (trained_model, training_history)
            
        Raises:
            ValueError: If model type is not supported
            RuntimeError: If training fails
        """
        try:
            print(f"Starting training with {model_type} architecture...")
            start_time = time.time()
            
            # Create model
            self.model = self._create_model(model_type)
            
            # Prepare data generators
            train_generator = self.data_pipeline.get_training_generator(self.config.batch_size)
            val_generator = self.data_pipeline.get_validation_generator(self.config.batch_size)
            
            # Prepare callbacks
            callbacks = self.callbacks_manager.get_all_callbacks()
            
            # Calculate steps per epoch
            train_steps = self._calculate_steps_per_epoch()
            val_steps = self._calculate_validation_steps()
            
            print(f"Training configuration:")
            print(f"  - Model: {model_type}")
            print(f"  - Epochs: {self.config.epochs}")
            print(f"  - Batch size: {self.config.batch_size}")
            print(f"  - Training steps per epoch: {train_steps}")
            print(f"  - Validation steps: {val_steps}")
            
            # Train the model
            print("Starting model training...")
            self.training_history = self.model.fit(
                train_generator,
                steps_per_epoch=train_steps,
                epochs=self.config.epochs,
                validation_data=val_generator,
                validation_steps=val_steps,
                callbacks=callbacks,
                verbose=self.config.verbose
            )
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Save model if requested
            if save_model:
                self._save_model()
            
            return self.model, self.training_history
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training process failed: {str(e)}") from e
    
    def _create_model(self, model_type: str) -> Model:
        """Create and compile model using factory."""
        try:
            model = self.model_factory.create_model(model_type, self.config)
            print(f"Model created successfully: {model.count_params():,} parameters")
            return model
        except Exception as e:
            raise ValueError(f"Failed to create {model_type} model: {str(e)}") from e
    
    def _calculate_steps_per_epoch(self) -> int:
        """Calculate training steps per epoch."""
        stats = self.data_pipeline.get_data_statistics()
        # Approximate patches per sample (you might want to make this more precise)
        avg_patches_per_sample = 19  # Based on your test results
        total_train_patches = stats['train_samples'] * avg_patches_per_sample
        steps = max(1, total_train_patches // self.config.batch_size)
        return steps
    
    def _calculate_validation_steps(self) -> int:
        """Calculate validation steps."""
        stats = self.data_pipeline.get_data_statistics()
        avg_patches_per_sample = 18  # Based on your test results
        total_val_patches = stats['val_samples'] * avg_patches_per_sample
        steps = max(1, total_val_patches // self.config.batch_size)
        return steps
    
    def _save_model(self):
        """Save the trained model."""
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")
        
        try:
            model_path = self.config.paths.models_dir / self.config.get_model_filename()
            self.model.save(model_path)
            print(f"Model saved to: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to save model: {str(e)}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process."""
        if self.training_history is None:
            return {"status": "not_trained"}
        
        history = self.training_history.history
        
        return {
            "status": "completed",
            "epochs_completed": len(history.get('loss', [])),
            "final_training_loss": history.get('loss', [])[-1] if history.get('loss') else None,
            "final_validation_loss": history.get('val_loss', [])[-1] if history.get('val_loss') else None,
            "best_validation_metric": max(history.get('val_jaccard_coeff', [0])) if history.get('val_jaccard_coeff') else None,
            "model_parameters": self.model.count_params() if self.model else None
        }