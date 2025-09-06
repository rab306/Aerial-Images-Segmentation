# main.py
import argparse
import sys
from pathlib import Path
import keras

from src.config.settings import Config
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.training.components import Evaluator, LossCalculator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train aerial image segmentation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--data_dir', 
        required=True, 
        help='Path to data directory containing Tile folders'
    )
    
    # Optional path arguments
    parser.add_argument(
        '--output_dir', 
        default='output', 
        help='Output directory for models and results'
    )
    parser.add_argument(
        '--model_path',
        help='Path to saved model file (required for --evaluate_only)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', 
        type=int, 
        help='Number of training epochs (overrides config default: 500)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        help='Batch size (overrides config default: 32)'
    )
    parser.add_argument(
        '--patch_size', 
        type=int, 
        help='Patch size (overrides config default: 256)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model_type', 
        default='unet', 
        choices=['unet'], 
        help='Model architecture'
    )
    
    # Actions
    parser.add_argument(
        '--evaluate_only', 
        action='store_true', 
        help='Only evaluate existing model, skip training'
    )
    parser.add_argument(
        '--no_save', 
        action='store_true', 
        help='Do not save the trained model'
    )
    parser.add_argument(
        '--dataset',
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset to evaluate on (default: test)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for training aerial image segmentation models."""
    print("=== Aerial Images Segmentation Training ===\n")
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Validate data directory exists
        if not Path(args.data_dir).exists():
            print(f"Error: Data directory does not exist: {args.data_dir}")
            sys.exit(1)
        
        # Create configuration with user-specified paths
        print("Loading configuration...")
        config = Config(data_dir=args.data_dir, output_dir=args.output_dir)
        
        # Override config parameters from command line
        if args.epochs:
            config.epochs = args.epochs
            print(f"Overriding epochs: {args.epochs}")
        
        if args.batch_size:
            config.batch_size = args.batch_size
            print(f"Overriding batch size: {args.batch_size}")
        
        if args.patch_size:
            config.patch_size = args.patch_size
            print(f"Overriding patch size: {args.patch_size}")
        
        # Print configuration summary
        config.print_summary()
        
        # Initialize components
        trainer = ModelTrainer(config)
        evaluator = ModelEvaluator(config)
        
        if not args.evaluate_only:
            # Train the model
            print(f"\nStarting training with {args.model_type} architecture...")
            model, history = trainer.train(
                model_type=args.model_type, 
                save_model=not args.no_save
            )
            
            # Print training summary
            summary = trainer.get_training_summary()
            print(f"\nTraining Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
            
            # Evaluate the trained model
            print(f"\nEvaluating trained model...")
            evaluator.evaluate_model(model, dataset="test")
            
        else:
            # Evaluation-only mode
            print(f"\nEvaluation-only mode: Loading pre-trained model...")
            
            # Validate model path
            if not args.model_path:
                print("Error: --model_path is required for evaluation-only mode")
                print("Usage: python main.py --data_dir <path> --evaluate_only --model_path <model_file>")
                sys.exit(1)
            
            model_path = Path(args.model_path)
            if not model_path.exists():
                print(f"Error: Model file not found: {model_path}")
                sys.exit(1)
            

            # Load the saved model
            try:
                print(f"Loading model from: {model_path}")

                # Load model without custom objects first
                model = keras.models.load_model(str(model_path), compile=False)
                
                # Create evaluator and loss calculator instances
                evaluator_metrics = Evaluator(config)
                loss_calc = LossCalculator(config)
                
                # Recompile the model with proper metrics
                model.compile(
                    optimizer='adam',
                    loss=loss_calc.combined_loss,
                    metrics=evaluator_metrics.get_metrics_list()
                )
                
                print(f"Model loaded and recompiled successfully:")
                print(f"  Architecture: {model.name}")
                print(f"  Parameters: {model.count_params():,}")
                print(f"  Input shape: {model.input_shape}")
                print(f"  Output shape: {model.output_shape}")

                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Make sure the model file is compatible and contains the required custom metrics")
                sys.exit(1)
            
            # Validate dataset choice
            dataset_choice = args.dataset
            available_datasets = ['train', 'val', 'test']
            if dataset_choice not in available_datasets:
                print(f"Error: Invalid dataset '{dataset_choice}'. Choose from: {available_datasets}")
                sys.exit(1)
            
            # Evaluate the loaded model
            print(f"\nEvaluating loaded model on {dataset_choice} dataset...")
            try:
                evaluator.evaluate_model(model, dataset=dataset_choice)
                
                # Print additional evaluation info
                print(f"\nModel Evaluation Completed:")
                print(f"  Model file: {model_path.name}")
                print(f"  Dataset: {dataset_choice}")
                print(f"  Results saved to: {config.paths.output_base}")
                
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                sys.exit(1)
        
        print(f"\n✅ Training and evaluation completed successfully!")
        print(f"Results saved to: {config.paths.output_base}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()