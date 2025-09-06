import time
import json
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from keras.models import Model
from keras.callbacks import History
from pathlib import Path

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
            
            # Prepare callbacks with model_type for proper file naming
            callbacks = self.callbacks_manager.get_all_callbacks(model_type=model_type)
            
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
            
            # Save training artifacts (plots, metrics, config)
            self._save_training_artifacts(model_type, training_time)
            
            # Save final model if requested
            if save_model:
                self._save_final_model(model_type)
            
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
    
    def _save_training_artifacts(self, model_type: str, training_time: float):
        """Save all training artifacts (plots, metrics, config)."""
        if self.training_history is None:
            print("Warning: No training history to save")
            return
        
        try:
            # Save training plots
            self._save_training_plots(model_type)
            
            # Save training metrics
            self._save_training_metrics(model_type, training_time)
            
            # Save configuration
            self._save_training_config()
            
        except Exception as e:
            print(f"Warning: Failed to save some training artifacts: {str(e)}")
    
    def _save_training_plots(self, model_type: str):
        """Save training plots to plots_dir."""
        history = self.training_history.history
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_type.upper()} Training History', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Jaccard coefficient plot
        if 'jaccard_coeff' in history:
            axes[0, 1].plot(history['jaccard_coeff'], label='Training Jaccard', color='blue')
            axes[0, 1].plot(history['val_jaccard_coeff'], label='Validation Jaccard', color='red')
            axes[0, 1].set_title('Jaccard Coefficient')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Jaccard Coefficient')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # IoU plot
        if 'one_hot_io_u' in history:
            axes[1, 0].plot(history['one_hot_io_u'], label='Training IoU', color='blue')
            axes[1, 0].plot(history['val_one_hot_io_u'], label='Validation IoU', color='red')
            axes[1, 0].set_title('IoU Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IoU')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Precision/Recall plot
        if 'precision' in history:
            axes[1, 1].plot(history['precision'], label='Training Precision', color='green')
            axes[1, 1].plot(history['recall'], label='Training Recall', color='orange')
            if 'val_precision' in history:
                axes[1, 1].plot(history['val_precision'], label='Val Precision', color='lightgreen')
                axes[1, 1].plot(history['val_recall'], label='Val Recall', color='coral')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to plots directory using config paths
        plot_path = self.config.paths.get_training_plot_path(f'{model_type}_training_history')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {plot_path}")
    
    def _save_training_metrics(self, model_type: str, training_time: float):
        """Save detailed training metrics to results_dir."""
        history = self.training_history.history
        
        # Calculate best metrics
        best_val_jaccard = max(history.get('val_jaccard_coeff', [0])) if history.get('val_jaccard_coeff') else 0
        best_val_loss = min(history.get('val_loss', [float('inf')])) if history.get('val_loss') else float('inf')
        
        # Find best epoch
        best_epoch = 0
        if 'val_jaccard_coeff' in history:
            best_epoch = history['val_jaccard_coeff'].index(best_val_jaccard) + 1
        
        metrics_data = {
            'model_info': {
                'model_type': model_type,
                'total_parameters': self.model.count_params() if self.model else 0,
                'model_architecture': str(self.model.name) if self.model else 'unknown'
            },
            'training_info': {
                'total_epochs': len(history.get('loss', [])),
                'training_time_seconds': round(training_time, 2),
                'best_epoch': best_epoch,
                'early_stopped': len(history.get('loss', [])) < self.config.epochs
            },
            'final_metrics': {
                'final_train_loss': history.get('loss', [])[-1] if history.get('loss') else None,
                'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
                'final_train_jaccard': history.get('jaccard_coeff', [])[-1] if history.get('jaccard_coeff') else None,
                'final_val_jaccard': history.get('val_jaccard_coeff', [])[-1] if history.get('val_jaccard_coeff') else None,
            },
            'best_metrics': {
                'best_val_jaccard': best_val_jaccard,
                'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
                'best_epoch': best_epoch
            },
            'training_config': {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'patch_size': self.config.patch_size,
                'num_classes': self.config.num_classes,
                'learning_rate': 'default',  # Add if you track this
                'optimizer': 'adam'  # Add if you track this
            },
            'data_info': self.data_pipeline.get_data_statistics(),
            'full_history': {
                key: [float(val) for val in values] 
                for key, values in history.items()
            }
        }
        
        # Save to results directory
        results_path = self.config.paths.results_dir / f'{model_type}_training_metrics.json'
        with open(results_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Training metrics saved to: {results_path}")
    
    def _save_training_config(self):
        """Save configuration used for this training run."""
        config_path = self.config.save_config()
        print(f"Training configuration saved to: {config_path}")
    
    def _save_final_model(self, model_type: str):
        """Save the final trained model to models_dir."""
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")
        
        try:
            # Use the improved path method
            model_path = self.config.paths.get_model_path(model_type)
            self.model.save(str(model_path))
            print(f"Final model saved to: {model_path}")
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
            "model_parameters": self.model.count_params() if self.model else None,
            "artifacts_saved": {
                "plots": str(self.config.paths.plots_dir),
                "metrics": str(self.config.paths.results_dir),
                "model": str(self.config.paths.models_dir),
                "logs": str(self.config.paths.logs_dir)
            }
        }