from typing import Dict, Type, List
from keras.models import Model

from src.models.base import BaseSegmentationModel
from src.models.architectures.Unet import UNetModel


class ModelFactory:
    """
    Factory class for creating semantic segmentation models.
    
    Provides centralized model creation with support for different architectures.
    Uses registry pattern to easily add new model types.
    """
    
    # Registry of available models
    _model_registry: Dict[str, Type[BaseSegmentationModel]] = {
        'unet': UNetModel,
        # Future models can be easily added here:
        # 'deeplab': DeepLabModel,
        # 'attention_unet': AttentionUNetModel,
    }
    
    @classmethod
    def create_model(cls, model_type: str, config, compile_model: bool = True) -> Model:
        """
        Create a model of the specified type.
        
        Args:
            model_type: Type of model to create ('unet', 'deeplab', etc.)
            config: Configuration object with model parameters
            compile_model: Whether to compile the model (default: True)
            
        Returns:
            Keras Model instance (compiled or uncompiled)
            
        Raises:
            ValueError: If model_type is not supported
            
        Example:
            >>> config = Config()
            >>> model = ModelFactory.create_model('unet', config)
            >>> print(model.summary())
        """
        if model_type not in cls._model_registry:
            available_models = ', '.join(cls.get_available_models())
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Available models: {available_models}"
            )
        
        # Get the model class and instantiate it
        model_class = cls._model_registry[model_type]
        model_builder = model_class()
        
        # Build the architecture
        model = model_builder.build_model(config)
        
        # Compile if requested
        if compile_model:
            model = model_builder.compile_model(model, config)
            
        print(f"Created {model_builder.get_model_name()} model with "
              f"{model.count_params():,} parameters")
        
        return model
    
    @classmethod
    def create_uncompiled_model(cls, model_type: str, config) -> Model:
        """
        Create an uncompiled model (useful for transfer learning).
        
        Args:
            model_type: Type of model to create
            config: Configuration object
            
        Returns:
            Uncompiled Keras Model
        """
        return cls.create_model(model_type, config, compile_model=False)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of all available model types.
        
        Returns:
            List of model type strings
        """
        return list(cls._model_registry.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict:
        """
        Get metadata about a specific model type.
        
        Args:
            model_type: Type of model to get info for
            
        Returns:
            Dictionary with model metadata
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._model_registry[model_type]
        model_builder = model_class()
        return model_builder.get_model_summary_info()
    
    @classmethod
    def register_model(cls, model_name: str, model_class: Type[BaseSegmentationModel]):
        """
        Register a new model type (for extending with custom models).
        
        Args:
            model_name: Name to register the model under
            model_class: Model class that implements BaseSegmentationModel
            
        Example:
            >>> ModelFactory.register_model('my_unet', MyCustomUNet)
            >>> model = ModelFactory.create_model('my_unet', config)
        """
        if not issubclass(model_class, BaseSegmentationModel):
            raise TypeError("Model class must inherit from BaseSegmentationModel")
        
        cls._model_registry[model_name] = model_class
        print(f"Registered new model type: '{model_name}'")
    
    @classmethod
    def list_models_with_info(cls) -> Dict[str, Dict]:
        """
        Get detailed information about all available models.
        
        Returns:
            Dictionary mapping model names to their metadata
        """
        models_info = {}
        for model_type in cls.get_available_models():
            try:
                models_info[model_type] = cls.get_model_info(model_type)
            except Exception as e:
                models_info[model_type] = {'error': str(e)}
        
        return models_info