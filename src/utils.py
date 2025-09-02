import tensorflow as tf
from keras.metrics import OneHotIoU, Recall, Precision
from keras.losses import CategoricalFocalCrossentropy, Dice
from keras.callbacks import ModelCheckpoint
from src.config import Config

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
            alpha=config.focal_alpha,  # From your config!
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


# Callbacks
def callbacks(checkpoint_path):
    """
    Creates a list of callbacks to be used during model training.

    Args:
        checkpoint_path (str): Path to save the model checkpoint.

    Returns:
        list: List of Keras callbacks.
    """

    callback_lst = []

    check_point = ModelCheckpoint(
        checkpoint_path,
        monitor='val_jaccard_coeff',
        save_weights_only=True,
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callback_lst.append(check_point)
    return callback_lst


