from abc import ABC, abstractmethod
from keras.models import Model
from typing import Dict, Any


class BaseSegmentationModel(ABC):
    """
    Abstract base class for all semantic segmentation models.
    
    This interface ensures all models have consistent methods for creation,
    compilation, and metadata access.
    """
    
    @abstractmethod
    def build_model(self, config) -> Model:
        """
        Build the model architecture.
        
        Args:
            config: Configuration object containing model parameters
            
        Returns:
            Keras Model instance (uncompiled)
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the model name for logging and identification.
        
        Returns:
            String identifier for this model type
        """
        pass
    
    @abstractmethod
    def get_default_compile_params(self) -> Dict[str, Any]:
        """
        Return default compilation parameters for this model.
        
        Returns:
            Dictionary with optimizer, loss, and metrics specifications
        """
        pass
    
    def compile_model(self, model: Model, config, custom_params: Dict = None) -> Model:
        """
        Compile the model with appropriate parameters.
        
        Args:
            model: Keras model to compile
            config: Configuration object
            custom_params: Optional custom compilation parameters
            
        Returns:
            Compiled Keras model
        """
        compile_params = self.get_default_compile_params()
        
        # Override with custom parameters if provided
        if custom_params:
            compile_params.update(custom_params)
        
        # Import components from your training module
        from satellite_segmentation.training.components import LossCalculator, Evaluator
        
        loss_calc = LossCalculator(config)
        evaluator = Evaluator(config)
        
        # Use your custom loss and metrics
        model.compile(
            optimizer=compile_params['optimizer'],
            loss=loss_calc.combined_loss,
            metrics=evaluator.get_metrics_list()
        )
        
        return model
    
    def get_model_summary_info(self) -> Dict[str, Any]:
        """
        Return metadata about the model architecture.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.get_model_name(),
            'type': 'semantic_segmentation',
            'framework': 'keras'
        }