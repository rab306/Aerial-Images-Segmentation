from Src.config import patch_size, num_channels, num_classes
from keras.models import Model
from keras.layers import (
    Input, 
    Conv2D, 
    BatchNormalization, 
    ReLU, 
    MaxPooling2D, 
    Conv2DTranspose, 
    concatenate
)
from keras.optimizers import Adam
from Src.utils import loss_func, evaluation_metrics

def custom_unet_model():
    """
    Builds a U-Net model for semantic segmentation.

    Returns:
    model (keras.Model): Compiled U-Net model.
    """

    # Input layer
    inputs = Input((patch_size, patch_size, num_channels))

    # Encoder path
    c0 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    c0 = BatchNormalization()(c0)
    c0 = ReLU()(c0)
    p0 = MaxPooling2D((2, 2))(c0)

    c1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(p0)
    c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(p1)
    c2 = BatchNormalization()(c2)
    c2 = ReLU()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(p2)
    c3 = BatchNormalization()(c3)
    c3 = ReLU()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(p3)
    c4 = BatchNormalization()(c4)
    c4 = ReLU()(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(p4)
    c5 = BatchNormalization()(c5)
    c5 = ReLU()(c5)
    p5 = MaxPooling2D((2, 2))(c5)

    # Bottleneck
    c6 = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(p5)
    c6 = BatchNormalization()(c6)
    c6 = ReLU()(c6)

    # Decoder path
    u7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(u7)
    c7 = BatchNormalization()(c7)
    c7 = ReLU()(c7)

    u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(u8)
    c8 = BatchNormalization()(c8)
    c8 = ReLU()(c8)

    u9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(u9)
    c9 = BatchNormalization()(c9)
    c9 = ReLU()(c9)

    u10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(u10)
    c10 = BatchNormalization()(c10)
    c10 = ReLU()(c10)

    u11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c1])
    c11 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(u11)
    c11 = BatchNormalization()(c11)
    c11 = ReLU()(c11)

    u12 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c11)
    u12 = concatenate([u12, c0])
    c12 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(u12)
    c12 = BatchNormalization()(c12)
    c12 = ReLU()(c12)

    # Output layer with softmax activation for multi-class segmentation
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c12)

    # Compile model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def compile_model():
    """
    Compiles the U-Net model with the specified loss function and metrics.

    Returns:
    model (keras.Model): Compiled U-Net model.
    """
    model = custom_unet_model()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss_func, metrics=evaluation_metrics())

    return model

