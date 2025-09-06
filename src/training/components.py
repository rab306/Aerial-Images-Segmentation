import tensorflow as tf
from keras.metrics import OneHotIoU, Recall, Precision
from keras.losses import CategoricalFocalCrossentropy, Dice
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from src.config.settings import Config
from typing import List

class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.num_classes = config.num_classes

    # Jaccard Coefficient Metric
    def jaccard_coeff(self, ground_truth, prediction):
        smooth = self.config.smooth
        intersection = tf.reduce_sum(ground_truth * prediction, axis=(-1))
        union = tf.reduce_sum(ground_truth + prediction, axis=(-1))
        jac = (intersection + smooth) / (union - intersection + smooth)
        return jac

    def get_metrics_list(self):
        class_ids = list(range(self.num_classes))
        metrics = [
            OneHotIoU(self.num_classes, target_class_ids=class_ids),
            Precision(),
            Recall(),
            self.jaccard_coeff
        ]
        return metrics

class LossCalculator:
    def __init__(self, config: Config):
        self.config = config
        # Configure once, use many times
        self.focal_loss = CategoricalFocalCrossentropy(
            alpha=config.focal_alpha,  
            gamma=config.focal_gamma
        )
        self.dice_loss = Dice()
    
    def combined_loss(self, ground_truth, prediction):
        focal = self.focal_loss(ground_truth, prediction)
        dice = self.dice_loss(ground_truth, prediction)
        return focal + dice
    
    # Individual losses for experimentation
    def focal_only(self, ground_truth, prediction):
        return self.focal_loss(ground_truth, prediction)
    
    def dice_only(self, ground_truth, prediction):
        return self.dice_loss(ground_truth, prediction)


class CallbacksManager:
    """Manages training callbacks for model training."""
    
    def __init__(self, config: Config):
        self.config = config
        self.primary_metric = config.primary_monitor_metric 
        self.patience = config.patience

    def get_checkpoint_callback(self, model_type: str = "unet"):
        """Get ModelCheckpoint callback using proper config paths."""
        checkpoint_path = self.config.paths.get_checkpoint_path(model_type)
        
        checkpt = ModelCheckpoint(
            str(checkpoint_path),  # Use the proper path from config
            monitor=self.primary_metric,  
            mode='max',
            save_weights_only=True,
            save_best_only=True,
            verbose=1
        )
        return checkpt
    
    def get_early_stopping_callback(self):  
        """Get EarlyStopping callback."""
        stop_point = EarlyStopping(
            patience=self.patience,  
            monitor=self.primary_metric,  
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        return stop_point
    
    def get_tensorboard_callback(self, model_type: str = "unet"):
        """Get TensorBoard callback for logging training metrics."""
        log_dir = self.config.paths.get_tensorboard_log_dir(model_type)
        
        tensorboard = TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,           # Log weight histograms every epoch
            write_graph=True,           # Log model graph
            write_images=True,          # Log model weights as images
            update_freq='epoch',        # Log metrics every epoch
            profile_batch=0             # Disable profiling for performance
        )
        return tensorboard
    
    def get_all_callbacks(self, model_type: str = "unet") -> List:
        """Get all configured callbacks for training."""
        return [
            self.get_checkpoint_callback(model_type),
            self.get_early_stopping_callback(),
            self.get_tensorboard_callback(model_type)
        ]
