# Description: Utility functions for the model.

# Defining Jaccard coefficient (Intersection over Union)
def jaccard_coeff(ground_truth, prediction):
    """
    Computes the Jaccard coefficient (Intersection over Union) between true and predicted masks.

    Args:
        ground_truth (tf.Tensor): Ground truth masks.
        prediction (tf.Tensor): Predicted masks.

    Returns:
        tf.Tensor: Jaccard coefficient.
    """
    import tensorflow as tf  
    from Src.config import smooth

    intersection = tf.reduce_sum(ground_truth * prediction, axis=(-1))
    union = tf.reduce_sum(ground_truth + prediction, axis=(-1))
    jac = (intersection + smooth) / (union - intersection + smooth)
    return jac

# Metrics for model evaluation
def evaluation_metrics(target_class_ids=None):
    """
    Returns a list of evaluation metrics for the model.

    Args:
        target_class_ids (list): List of target class IDs for the OneHotIoU metric.

    Returns:
        list: List of metrics to be used during model evaluation.
    """
    from Src.config import num_classes
    from keras.metrics import OneHotIoU, Recall, Precision
    if target_class_ids is None:
        target_class_ids = list(range(num_classes))  # Default to all classes
    metrics = [
        OneHotIoU(num_classes, target_class_ids=target_class_ids),
        Precision(),
        Recall(),
        jaccard_coeff
    ]
    return metrics

# Loss function
def loss_func(ground_truth, prediction):
    """
    Computes a combined loss function consisting of Dice loss and Focal loss.

    Args:
        ground_truth (tf.Tensor): Ground truth masks.
        prediction (tf.Tensor): Predicted masks.

    Returns:
        tf.Tensor: Total combined loss.
    """
    from keras.losses import CategoricalFocalCrossentropy, Dice

    focal = CategoricalFocalCrossentropy(alpha=0.5, gamma=1)
    dice = Dice()
    dice_loss = dice(ground_truth, prediction)
    focal_loss = focal(ground_truth, prediction)
    total_loss = focal_loss + dice_loss
    return total_loss

# Callbacks
def callbacks(checkpoint_path):
    """
    Creates a list of callbacks to be used during model training.

    Args:
        checkpoint_path (str): Path to save the model checkpoint.

    Returns:
        list: List of Keras callbacks.
    """

    from keras.callbacks import ModelCheckpoint
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


