from Src.config import seed 

def split_data():
    """
    Split the data into training, validation, and testing sets.

    Returns:
        Tuple: Training, validation, and testing sets for images and labels.
    """

    from Src.data.data_loader import load_images
    from Src.data.preprocess_data import one_hot_encoding

    images = load_images()
    labels = one_hot_encoding()

    from sklearn.model_selection import train_test_split
    X_train, X_val_test, y_train, y_val_test = train_test_split(images, labels, test_size=0.20, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.50, random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test


def data_generator():
    """
    Data generator for training the model.

    Yields:
        Tuple: Batch of images and labels
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from Src.config import data_gen_args, batch_size

    image_data_gen = ImageDataGenerator(**data_gen_args)
    X_train, y_train, _, _, _, _ = split_data()
    image_generator = image_data_gen.flow(X_train, batch_size=batch_size, seed=seed)
    label_generator = image_data_gen.flow(y_train, batch_size=batch_size, seed=seed)

    while True:
        # Get the next batch of images and labels
        X_batch = next(image_generator)
        y_batch = next(label_generator)

        yield X_batch, y_batch



