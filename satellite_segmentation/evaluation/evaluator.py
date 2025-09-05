import numpy as np
from typing import Dict, List, Tuple, Optional
from keras.models import Model

from satellite_segmentation.config.settings import Config
from satellite_segmentation.training.pipeline import TrainingDataPipeline


class ModelEvaluator:
    """
    Comprehensive model evaluation with detailed metrics and analysis.
    
    Provides both basic evaluation metrics and detailed per-class analysis
    for semantic segmentation models.
    """
    
    def __init__(self, config: Config):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_pipeline = TrainingDataPipeline(config)
        
    def evaluate_model(self, model: Model, dataset: str = "test") -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained Keras model to evaluate
            dataset: Dataset to evaluate on ("test", "val", or "train")
            
        Returns:
            Dictionary containing evaluation results
            
        Raises:
            ValueError: If dataset is not supported
            RuntimeError: If evaluation fails
        """
        try:
            print(f"Evaluating model on {dataset} dataset...")
            
            # Get data
            X_data, y_data = self._get_dataset(dataset)
            
            if X_data is None or y_data is None:
                raise ValueError(f"Could not load {dataset} data for evaluation")
            
            print(f"Evaluating on {len(X_data)} samples...")
            
            # Basic evaluation using model.evaluate()
            basic_metrics = self._evaluate_basic_metrics(model, X_data, y_data)
            
            # Detailed evaluation with predictions
            detailed_metrics = self._evaluate_detailed_metrics(model, X_data, y_data)
            
            # Combine results
            evaluation_results = {
                "dataset": dataset,
                "sample_count": len(X_data),
                "basic_metrics": basic_metrics,
                "detailed_metrics": detailed_metrics,
                "class_names": self.config.get_class_names()
            }
            
            self._print_evaluation_summary(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            raise RuntimeError(f"Model evaluation failed: {str(e)}") from e
    
    def _get_dataset(self, dataset: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the specified dataset."""
        dataset_getters = {
            "test": self.data_pipeline.get_test_data,
            "val": self.data_pipeline.get_val_data,
            "train": self.data_pipeline.get_train_data
        }
        
        if dataset not in dataset_getters:
            raise ValueError(f"Unsupported dataset: {dataset}. Choose from: {list(dataset_getters.keys())}")
        
        try:
            return dataset_getters[dataset]()
        except Exception as e:
            print(f"Failed to load {dataset} dataset: {str(e)}")
            return None, None
    
    def _evaluate_basic_metrics(self, model: Model, X_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Evaluate using model.evaluate() method."""
        try:
            results = model.evaluate(
                X_data, y_data, 
                batch_size=self.config.batch_size, 
                verbose=self.config.verbose
            )
            
            # Map results to metric names
            metric_names = ['loss'] + [m.name for m in model.metrics]
            
            return dict(zip(metric_names, results))
            
        except Exception as e:
            print(f"Basic evaluation failed: {str(e)}")
            return {"error": str(e)}
    
    def _evaluate_detailed_metrics(self, model: Model, X_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Evaluate with detailed per-class metrics."""
        try:
            # Get predictions
            print("Generating predictions...")
            y_pred = model.predict(X_data, batch_size=self.config.batch_size, verbose=0)
            
            # Convert one-hot to class indices
            y_true_classes = np.argmax(y_data, axis=-1)
            y_pred_classes = np.argmax(y_pred, axis=-1)
            
            # Calculate per-class metrics
            per_class_metrics = self._calculate_per_class_metrics(y_true_classes, y_pred_classes)
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(y_true_classes, y_pred_classes)
            
            return {
                "per_class": per_class_metrics,
                "overall": overall_metrics
            }
            
        except Exception as e:
            print(f"Detailed evaluation failed: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate precision, recall, F1 for each class."""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true.flatten(), y_pred.flatten(), 
            labels=range(self.config.num_classes),
            average=None,
            zero_division=0
        )
        
        class_names = self.config.get_class_names()
        per_class = {}
        
        for i, class_name in enumerate(class_names):
            per_class[class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }
        
        return per_class
    
    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate overall metrics."""
        from sklearn.metrics import accuracy_score, jaccard_score
        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        accuracy = accuracy_score(y_true_flat, y_pred_flat)
        jaccard = jaccard_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
        
        return {
            "accuracy": float(accuracy),
            "jaccard_score": float(jaccard),
            "total_pixels": len(y_true_flat)
        }
    
    def _print_evaluation_summary(self, results: Dict):
        """Print a formatted evaluation summary."""
        print("\n" + "="*50)
        print(f"EVALUATION SUMMARY - {results['dataset'].upper()} DATASET")
        print("="*50)
        
        # Basic metrics
        if "basic_metrics" in results and "error" not in results["basic_metrics"]:
            print("\nBasic Metrics:")
            for metric, value in results["basic_metrics"].items():
                print(f"  {metric}: {value:.4f}")
        
        # Overall metrics
        if "detailed_metrics" in results and "overall" in results["detailed_metrics"]:
            overall = results["detailed_metrics"]["overall"]
            print(f"\nOverall Performance:")
            print(f"  Accuracy: {overall['accuracy']:.4f}")
            print(f"  Jaccard Score: {overall['jaccard_score']:.4f}")
        
        # Per-class summary
        if "detailed_metrics" in results and "per_class" in results["detailed_metrics"]:
            print(f"\nPer-Class F1 Scores:")
            per_class = results["detailed_metrics"]["per_class"]
            for class_name, metrics in per_class.items():
                print(f"  {class_name}: {metrics['f1_score']:.4f}")
        
        print("="*50)