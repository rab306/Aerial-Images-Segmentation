import numpy as np
import cv2
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from src.config.settings import Config
from src.training.components import Evaluator, LossCalculator


class ModelPredictor:
    """Predictor class for running inference with trained segmentation models."""
    
    def __init__(self, model: keras.Model, config: Config):
        """
        Initialize predictor with loaded model and configuration.
        
        Args:
            model: Loaded Keras model
            config: Configuration object
        """
        self.model = model
        self.config = config
        
        # Model metadata
        self.input_shape = model.input_shape[1:3]  # (height, width)
        self.num_classes = model.output_shape[-1]
        self.patch_size = config.patch_size
        
        print(f"Predictor initialized:")
        print(f"  Model parameters: {self.model.count_params():,}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Patch size: {self.patch_size}")
    
    @classmethod
    def load_from_file(cls, model_path: str, config: Config) -> 'ModelPredictor':
        """
        Load predictor from saved model file.
        
        Args:
            model_path: Path to saved model (.h5 or .keras)
            config: Configuration object
            
        Returns:
            ModelPredictor instance
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        try:
            # Load model without custom objects first
            model = keras.models.load_model(str(model_path), compile=False)
            
            # Recompile with proper metrics for consistency
            evaluator_metrics = Evaluator(config)
            loss_calc = LossCalculator(config)
            
            model.compile(
                optimizer='adam',
                loss=loss_calc.combined_loss,
                metrics=evaluator_metrics.get_metrics_list()
            )
            
            print(f"Model loaded successfully: {model.name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
        
        return cls(model, config)
    
    def predict_single_image(self, image_path: str, overlap_ratio: float = 0.1) -> np.ndarray:
        """
        Predict segmentation mask for a single image.
        
        Args:
            image_path: Path to input image
            overlap_ratio: Overlap ratio for patch-based prediction (0.0-0.5)
            
        Returns:
            Predicted segmentation mask as numpy array
        """
        # Load and validate image
        image = self._load_image(image_path)
        
        # Generate patches with overlap
        patches, patch_info = self._create_overlapping_patches(image, overlap_ratio)
        
        # Run prediction on patches
        print(f"Running inference on {len(patches)} patches...")
        patch_predictions = self.model.predict(patches, verbose=0)
        
        # Reconstruct full image prediction
        full_prediction = self._reconstruct_from_patches(
            patch_predictions, patch_info, image.shape[:2]
        )
        
        return full_prediction
    
    def predict_batch(self, image_paths: List[str], output_dir: str, 
                 overlap_ratio: float = 0.1, save_confidence: bool = False,
                 save_visualizations: bool = False, comprehensive_viz: bool = True) -> List[Dict[str, Any]]:
        """
        Predict segmentation masks for multiple images.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save predictions
            overlap_ratio: Overlap ratio for patches
            save_confidence: Whether to save confidence maps
            save_visualizations: Whether to save visualization overlays
            comprehensive_viz: Whether to use comprehensive 3-panel visualization
            
        Returns:
            List of prediction results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                # Run prediction
                prediction = self.predict_single_image(image_path, overlap_ratio)
                
                # Generate output filename
                input_name = Path(image_path).stem
                output_file = output_path / f"{input_name}_prediction.png"
                
                # Save class prediction
                self.save_class_prediction(prediction, str(output_file))
                
                result = {
                    'input_path': image_path,
                    'output_path': str(output_file),
                    'success': True,
                    'error': None
                }
                
                # Save confidence maps if requested
                if save_confidence:
                    confidence_file = output_path / f"{input_name}_confidence.npy"
                    np.save(confidence_file, prediction)
                    result['confidence_path'] = str(confidence_file)
                
                # Save visualizations if requested
                if save_visualizations:
                    if comprehensive_viz:
                        # Use new comprehensive visualization
                        viz_file = output_path / f"{input_name}_comprehensive.png"
                        self.save_comprehensive_visualization(image_path, prediction, str(viz_file))
                        result['comprehensive_viz_path'] = str(viz_file)
                    else:
                        # Use simple overlay
                        viz_file = output_path / f"{input_name}_overlay.png"
                        self._save_visualization_overlay(image_path, prediction, str(viz_file))
                        result['visualization_path'] = str(viz_file)
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'input_path': image_path,
                    'output_path': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and validate image from file."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            print(f"Loaded image: {image.shape}")
            return image
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")
    
    def _create_overlapping_patches(self, image: np.ndarray, 
                                   overlap_ratio: float) -> Tuple[np.ndarray, Dict]:
        """
        Create overlapping patches from image for better prediction quality.
        
        Args:
            image: Input image
            overlap_ratio: Overlap between patches (0.0-0.5)
            
        Returns:
            Tuple of (patches_array, patch_info_dict)
        """
        h, w = image.shape[:2]
        patch_size = self.patch_size
        
        # Calculate step size (overlap)
        step = int(patch_size * (1 - overlap_ratio))
        
        patches = []
        patch_coords = []
        
        # Generate patches with overlap
        y_positions = list(range(0, h - patch_size + 1, step))
        x_positions = list(range(0, w - patch_size + 1, step))
        
        # Ensure we cover the entire image
        if y_positions[-1] + patch_size < h:
            y_positions.append(h - patch_size)
        if x_positions[-1] + patch_size < w:
            x_positions.append(w - patch_size)
        
        for y in y_positions:
            for x in x_positions:
                # Extract patch
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # Normalize patch (same as training preprocessing)
                patch_normalized = patch.astype(np.float32) / 255.0
                
                patches.append(patch_normalized)
                patch_coords.append((y, x))
        
        return np.array(patches), {
            'coords': patch_coords,
            'original_shape': (h, w),
            'patch_size': patch_size,
            'overlap_ratio': overlap_ratio
        }
    
    def _reconstruct_from_patches(self, patch_predictions: np.ndarray, 
                                 patch_info: Dict, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct full image prediction from overlapping patches.
        Uses averaging for overlapping regions.
        """
        h, w = original_shape
        num_classes = patch_predictions.shape[-1]
        patch_size = patch_info['patch_size']
        
        # Initialize output and weight arrays
        prediction_sum = np.zeros((h, w, num_classes), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        
        # Accumulate predictions with weights
        for pred, (y, x) in zip(patch_predictions, patch_info['coords']):
            # Add prediction to sum
            prediction_sum[y:y+patch_size, x:x+patch_size] += pred
            
            # Add weight (can be uniform or distance-based)
            weight_sum[y:y+patch_size, x:x+patch_size] += 1.0
        
        # Average overlapping predictions
        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-8)
        
        # Normalize by weights
        final_prediction = prediction_sum / weight_sum[..., np.newaxis]
        
        return final_prediction

    def save_comprehensive_visualization(self, original_image_path: str, prediction: np.ndarray, 
                                    output_path: str):
        """
        Save comprehensive visualization with original image, prediction mask, and overlay.
        
        Args:
            original_image_path: Path to original input image
            prediction: Model prediction (H, W, num_classes)
            output_path: Path to save the visualization
        """
        # Load original image
        original = self._load_image(original_image_path)
        
        # Convert prediction to class indices and apply colors
        class_pred = np.argmax(prediction, axis=-1)
        colored_pred = self._apply_class_colors(class_pred)
        
        # Resize prediction to match original if needed
        if colored_pred.shape[:2] != original.shape[:2]:
            colored_pred = cv2.resize(
                colored_pred, 
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Create overlay (60% original, 40% prediction)
        overlay = cv2.addWeighted(original, 0.6, colored_pred, 0.4, 0)
        
        # Create matplotlib figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Prediction mask
        axes[1].imshow(colored_pred)
        axes[1].set_title('Prediction Mask', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Original + Prediction)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Add class legend
        self._add_class_legend(fig)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Comprehensive visualization saved: {output_path}")

    def _add_class_legend(self, fig):
        """Add class color legend to the figure."""
        class_colors = [
            [60, 16, 152],    # Building - Purple
            [132, 41, 246],   # Land - Light Purple  
            [110, 193, 228],  # Road - Light Blue
            [254, 221, 58],   # Vegetation - Yellow
            [226, 169, 41],   # Water - Orange
            [155, 155, 155],  # Unlabeled - Gray
        ]
        
        class_names = self.config.get_class_names()
        
        # Normalize colors to [0,1] for matplotlib
        colors_normalized = [[c/255.0 for c in color] for color in class_colors]
        
        # Create legend patches
        legend_elements = []
        for i, (name, color) in enumerate(zip(class_names, colors_normalized)):
            if i < len(colors_normalized):
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                )
        
        # Add legend to the figure
        fig.legend(legend_elements, class_names, 
                loc='center', bbox_to_anchor=(0.5, 0.02), 
                ncol=len(class_names), fontsize=10,
                title='Class Legend', title_fontsize=12)

    def save_class_prediction(self, prediction: np.ndarray, output_path: str):
        """Save prediction as colored class image."""
        # Convert to class indices
        class_prediction = np.argmax(prediction, axis=-1)
        
        # Apply class colors
        colored_prediction = self._apply_class_colors(class_prediction)
        
        # Save as PNG
        Image.fromarray(colored_prediction).save(output_path)
        print(f"Class prediction saved: {output_path}")

    def _apply_class_colors(self, class_prediction: np.ndarray) -> np.ndarray:
        """Apply class colors to prediction mask."""
        # Define colors for each class (RGB) - matching dataset colors
        class_colors = [
            [60, 16, 152],    # Building - Purple (#3C1098)
            [132, 41, 246],   # Land - Light Purple (#8429F6)
            [110, 193, 228],  # Road - Light Blue (#6EC1E4)
            [254, 221, 58],   # Vegetation - Yellow (#FEDD3A)
            [226, 169, 41],   # Water - Orange (#E2A929)
            [155, 155, 155],  # Unlabeled - Gray (#9B9B9B)
        ]
        
        h, w = class_prediction.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx, color in enumerate(class_colors):
            if class_idx < self.num_classes:
                mask = class_prediction == class_idx
                colored[mask] = color
        
        return colored

    # Keep the old overlay function for backward compatibility
    def _save_visualization_overlay(self, original_image_path: str, 
                                prediction: np.ndarray, output_path: str):
        """Save simple visualization overlay of original image and prediction."""
        # Load original image
        original = self._load_image(original_image_path)
        
        # Convert prediction to class indices
        class_pred = np.argmax(prediction, axis=-1)
        colored_pred = self._apply_class_colors(class_pred)
        
        # Resize prediction to match original if needed
        if colored_pred.shape[:2] != original.shape[:2]:
            colored_pred = cv2.resize(
                colored_pred, 
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Create overlay (60% original, 40% prediction)
        overlay = cv2.addWeighted(original, 0.6, colored_pred, 0.4, 0)
        
        # Save
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, overlay_bgr)
        print(f"Visualization overlay saved: {output_path}")