#!/usr/bin/env python3
"""
Inference script for aerial image segmentation.
Use this to run predictions on new images using trained models.
"""
import argparse
import sys
from pathlib import Path
import time

from src.config.settings import Config
from src.inference.predictor import ModelPredictor


def parse_arguments():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(
        description='Run inference on aerial images using trained models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model_path', 
        required=True, 
        help='Path to trained model file (.h5 or .keras)'
    )
    parser.add_argument(
        '--input_path', 
        required=True, 
        help='Path to input image or directory of images'
    )
    parser.add_argument(
        '--output_path', 
        required=True, 
        help='Output directory for predictions'
    )
    
    # Optional processing arguments
    parser.add_argument(
        '--patch_size', 
        type=int, 
        default=256,
        help='Patch size for processing (should match training)'
    )
    parser.add_argument(
        '--overlap_ratio', 
        type=float, 
        default=0.1,
        help='Overlap ratio between patches (0.0-0.5, higher = better quality, slower)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='Batch size for processing multiple images'
    )
    
    # Output options
    parser.add_argument(
        '--save_confidence',
        action='store_true',
        help='Save confidence maps (.npy files) along with predictions'
    )
    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='Save visualization overlays of original + prediction'
    )
    parser.add_argument(
        '--comprehensive_viz',
        action='store_true',
        help='Use comprehensive 3-panel visualization (original, mask, overlay) instead of simple overlay'
    )

    # Processing options
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.tif', '.tiff'],
        help='Image file extensions to process (when input is directory)'
    )
    
    return parser.parse_args()


def find_images(input_path: Path, extensions: list) -> list:
    """Find all image files in directory with specified extensions."""
    if input_path.is_file():
        return [input_path]
    
    image_files = []
    extensions_lower = [ext.lower() for ext in extensions]
    
    for ext in extensions_lower:
        image_files.extend(input_path.rglob(f'*{ext}'))
    
    return sorted(image_files)


def print_processing_summary(results: list, processing_time: float):
    """Print summary of processing results."""
    total_images = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_images - successful
    
    print(f"\n{'='*50}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Average time per image: {processing_time/max(total_images, 1):.1f} seconds")
    
    if failed > 0:
        print(f"\nFailed images:")
        for result in results:
            if not result['success']:
                print(f"  {Path(result['input_path']).name}: {result['error']}")
    
    print(f"{'='*50}")


def main():
    """Main inference function."""
    print("=== Aerial Images Segmentation - Inference ===\n")
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate input paths
        model_path = Path(args.model_path)
        input_path = Path(args.input_path)
        output_path = Path(args.output_path)
        
        if not model_path.exists():
            print(f"Error: Model file does not exist: {model_path}")
            sys.exit(1)
            
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            sys.exit(1)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create configuration (minimal for inference)
        config = Config(inference_mode=True)
        config.patch_size = args.patch_size
        
        # Load model and create predictor
        print(f"Loading model: {model_path.name}")
        start_time = time.time()
        
        predictor = ModelPredictor.load_from_file(str(model_path), config)
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.1f} seconds\n")
        
        # Find images to process
        image_files = find_images(input_path, args.extensions)
        
        if not image_files:
            print(f"No image files found in: {input_path}")
            print(f"Looking for extensions: {args.extensions}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} image(s) to process")
        print(f"Output directory: {output_path}")
        print(f"Overlap ratio: {args.overlap_ratio}")
        
        if args.save_confidence:
            print("Will save confidence maps")
        if args.save_visualizations:
            print("Will save visualization overlays")
        
        print(f"\nStarting inference...\n")
        
        # Process images
        processing_start = time.time()
        
        if len(image_files) == 1:
            # Single image processing
            image_path = str(image_files[0])
            print(f"Processing single image: {Path(image_path).name}")
            
            try:
                prediction = predictor.predict_single_image(image_path, args.overlap_ratio)
                
                # Save results
                output_file = output_path / f"{Path(image_path).stem}_prediction.png"
                predictor.save_class_prediction(prediction, str(output_file))
                
                results = [{
                    'input_path': image_path,
                    'output_path': str(output_file),
                    'success': True,
                    'error': None
                }]
                
                # Save optional outputs
                if args.save_confidence:
                    confidence_file = output_path / f"{Path(image_path).stem}_confidence.npy"
                    import numpy as np
                    np.save(confidence_file, prediction)
                    print(f"Confidence map saved: {confidence_file}")
                
                if args.comprehensive_viz:
                    # Use comprehensive visualization
                    viz_file = output_path / f"{Path(image_path).stem}_comprehensive.png"
                    predictor.save_comprehensive_visualization(image_path, prediction, str(viz_file))
                    print(f"Comprehensive visualization saved: {viz_file}")
                else:
                    # Use simple overlay
                    viz_file = output_path / f"{Path(image_path).stem}_overlay.png"
                    predictor._save_visualization_overlay(image_path, prediction, str(viz_file))
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results = [{
                    'input_path': image_path,
                    'output_path': None,
                    'success': False,
                    'error': str(e)
                }]
        
        else:
            # Batch processing
            print(f"Processing {len(image_files)} images...")
            
            results = predictor.predict_batch(
                [str(f) for f in image_files],
                str(output_path),
                overlap_ratio=args.overlap_ratio,
                save_confidence=args.save_confidence,
                save_visualizations=args.save_visualizations,
                comprehensive_viz=args.comprehensive_viz  # Add this line
            )
        
        processing_time = time.time() - processing_start
        
        # Print summary
        print_processing_summary(results, processing_time)
        
        # Success message
        successful_count = sum(1 for r in results if r['success'])
        if successful_count > 0:
            print(f"\n✅ Successfully processed {successful_count} image(s)!")
            print(f"Results saved to: {output_path}")
        else:
            print(f"\n❌ No images were processed successfully.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()