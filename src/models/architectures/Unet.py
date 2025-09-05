from keras.models import Model
from keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, 
    Conv2DTranspose, concatenate
)
from keras.optimizers import Adam
from typing import Dict, Any

from src.models.base import BaseSegmentationModel


class UNetModel(BaseSegmentationModel):
    """
    U-Net architecture implementation for semantic segmentation.
    
    Classic U-Net with encoder-decoder structure and skip connections.
    """
    
    def build_model(self, config) -> Model:
        """
        Build U-Net model architecture.
        
        Args:
            config: Configuration object with patch_size, num_channels, num_classes
            
        Returns:
            Uncompiled Keras Model
        """
        # Input layer
        inputs = Input((config.patch_size, config.patch_size, config.num_channels))
        
        # Encoder path
        c0, p0 = self._encoder_block(inputs, 16, "enc_0")
        c1, p1 = self._encoder_block(p0, 32, "enc_1") 
        c2, p2 = self._encoder_block(p1, 64, "enc_2")
        c3, p3 = self._encoder_block(p2, 128, "enc_3")
        c4, p4 = self._encoder_block(p3, 256, "enc_4")
        c5, p5 = self._encoder_block(p4, 512, "enc_5")
        
        # Bottleneck
        c6 = self._conv_block(p5, 1024, "bottleneck")
        
        # Decoder path
        c7 = self._decoder_block(c6, c5, 512, "dec_7")
        c8 = self._decoder_block(c7, c4, 256, "dec_8")
        c9 = self._decoder_block(c8, c3, 128, "dec_9")
        c10 = self._decoder_block(c9, c2, 64, "dec_10")
        c11 = self._decoder_block(c10, c1, 32, "dec_11")
        c12 = self._decoder_block(c11, c0, 16, "dec_12")
        
        # Output layer
        outputs = Conv2D(
            config.num_classes, 
            (1, 1), 
            activation='softmax',
            name="output"
        )(c12)
        
        model = Model(inputs=[inputs], outputs=[outputs], name="UNet")
        return model
    
    def _encoder_block(self, inputs, filters, name_prefix):
        """
        Create encoder block with convolution + pooling.
        
        Args:
            inputs: Input tensor
            filters: Number of filters
            name_prefix: Prefix for layer names
            
        Returns:
            Tuple of (conv_output, pooled_output)
        """
        conv = self._conv_block(inputs, filters, name_prefix)
        pool = MaxPooling2D((2, 2), name=f"{name_prefix}_pool")(conv)
        return conv, pool
    
    def _decoder_block(self, inputs, skip_connection, filters, name_prefix):
        """
        Create decoder block with upsampling + skip connection.
        
        Args:
            inputs: Input tensor from previous layer
            skip_connection: Skip connection from encoder
            filters: Number of filters
            name_prefix: Prefix for layer names
            
        Returns:
            Output tensor
        """
        upsample = Conv2DTranspose(
            filters, (2, 2), 
            strides=(2, 2), 
            padding='same',
            name=f"{name_prefix}_upsample"
        )(inputs)
        
        concat = concatenate([upsample, skip_connection], name=f"{name_prefix}_concat")
        conv = self._conv_block(concat, filters, name_prefix)
        return conv
    
    def _conv_block(self, inputs, filters, name_prefix):
        """
        Create convolution block with BatchNorm + ReLU.
        
        Args:
            inputs: Input tensor
            filters: Number of filters
            name_prefix: Prefix for layer names
            
        Returns:
            Output tensor
        """
        conv = Conv2D(
            filters, (3, 3), 
            padding='same', 
            kernel_initializer='he_normal',
            name=f"{name_prefix}_conv"
        )(inputs)
        norm = BatchNormalization(name=f"{name_prefix}_bn")(conv)
        activation = ReLU(name=f"{name_prefix}_relu")(norm)
        return activation
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return "UNet"
    
    def get_default_compile_params(self) -> Dict[str, Any]:
        """Return default compilation parameters for U-Net."""
        return {
            'optimizer': Adam(learning_rate=1e-4),
        }
    
    def get_model_summary_info(self) -> Dict[str, Any]:
        """Return U-Net specific metadata."""
        info = super().get_model_summary_info()
        info.update({
            'architecture': 'UNet',
            'encoder_blocks': 6,
            'decoder_blocks': 6,
            'skip_connections': True,
            'description': 'Classic U-Net with encoder-decoder structure'
        })
        return info