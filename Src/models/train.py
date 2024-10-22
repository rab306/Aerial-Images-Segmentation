from Src.config import batch_size, epochs, checkpoint_path, model_path
from Src.features.features import split_data, data_generator   
from Src.utils import callbacks as create_callbacks
from Src.models.Unet_model import compile_model


def train_model():
    """
    Train the U-Net model.

    Returns:
        Tuple: Trained model and training history.
    """
    try:
        X_train, _, X_val, y_val, _, _ = split_data()
        train_generator = data_generator()
        training_callbacks = create_callbacks(checkpoint_path)
        step_per_epoch = len(X_train) // batch_size
        model = compile_model()

        # Train the model
        history = model.fit(
            train_generator,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=training_callbacks,
            steps_per_epoch=step_per_epoch
        )

        model.save(model_path)

        return model, history

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


        

