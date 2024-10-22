from Src.features.features import split_data

def evaluate_model(model):
    """
    Evaluate the model using the provided test data.

    Parameters:
        model (keras.Model): Keras model to be evaluated.

    Returns:
        results (list): List of evaluation metrics for the model.
    """
    from Src.config import batch_size, verbose
    
    try:
        _, _, _, _, X_test, y_test = split_data()
        
        if X_test is None or y_test is None:
            raise ValueError("Test data could not be loaded.")
        
        # Evaluate the model on the test data
        results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
        return results
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None

