#!/usr/bin/env python3
"""
Example script demonstrating the DINOv2 + TokenCut Pipeline

This creates sample data and runs the pipeline.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2


def create_sample_dataset(output_dir):
    """
    Create a sample dataset with synthetic images.
    
    Args:
        output_dir: Directory to save sample images
    """
    output_path = Path(output_dir)
    dataset_path = output_path / 'sample_dataset' / 'objects'
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating sample dataset in: {dataset_path}")
    
    # Create 3 sample images with different objects
    images = [
        # Image 1: Red rectangle
        ('rectangle.jpg', create_rectangle_image()),
        # Image 2: Blue circle
        ('circle.jpg', create_circle_image()),
        # Image 3: Green triangle
        ('triangle.jpg', create_triangle_image()),
    ]
    
    for filename, image in images:
        filepath = dataset_path / filename
        cv2.imwrite(str(filepath), image)
        print(f"  Created: {filename}")
    
    return dataset_path.parent


def create_rectangle_image(size=(480, 640)):
    """Create an image with a red rectangle."""
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 240
    
    h, w = size
    # Draw red rectangle
    cv2.rectangle(image, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 200), -1)
    
    return image


def create_circle_image(size=(480, 640)):
    """Create an image with a blue circle."""
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 240
    
    h, w = size
    # Draw blue circle
    cv2.circle(image, (w//2, h//2), min(h, w)//4, (200, 0, 0), -1)
    
    return image


def create_triangle_image(size=(480, 640)):
    """Create an image with a green triangle."""
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 240
    
    h, w = size
    # Define triangle points
    pts = np.array([
        [w//2, h//4],
        [w//4, 3*h//4],
        [3*w//4, 3*h//4]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw green triangle
    cv2.fillPoly(image, [pts], (0, 200, 0))
    
    return image


def run_example_pipeline():
    """Run the pipeline on sample data."""
    print("\n" + "=" * 70)
    print("DINOv2 + TokenCut Pipeline - Example Demonstration")
    print("=" * 70 + "\n")
    
    # Create temporary directory for this example
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create sample dataset
        dataset_path = create_sample_dataset(tmpdir)
        input_dir = dataset_path / 'objects'
        output_dir = tmpdir / 'results'
        
        print(f"\nInput directory: {input_dir}")
        print(f"Output directory: {output_dir}\n")
        
        # Import the pipeline
        try:
            import dinov2_tokencut_pipeline as pipeline
        except ImportError:
            print("Error: Could not import dinov2_tokencut_pipeline.py")
            print("Make sure the script is in the same directory.")
            return 1
        
        print("Running pipeline with arguments:")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Model: dinov2_vits14 (test mode)")
        print(f"  Threshold (tau): 0.2")
        print("\n" + "-" * 70 + "\n")
        
        try:
            # Run the pipeline with explicit arguments
            # Temporarily save and restore sys.argv to avoid side effects
            original_argv = sys.argv.copy()
            try:
                sys.argv = [
                    'example',
                    '--input_dir', str(input_dir),
                    '--output_dir', str(output_dir),
                    '--model_name', 'dinov2_vits14',
                    '--tau', '0.2',
                    '--log_level', 'INFO',
                    '--continue_on_error',
                    '--test_mode'
                ]
                
                args = pipeline.parse_arguments()
                pipe = pipeline.ObjectLocalizationPipeline(args)
                pipe.run()
            finally:
                sys.argv = original_argv
            
            print("\n" + "-" * 70)
            print("\n✓ Pipeline completed successfully!\n")
            
            # Show results
            print("Results saved to:")
            print(f"  Cutouts: {output_dir / 'cutouts'}")
            print(f"  Bounding boxes: {output_dir / 'bboxes'}")
            print(f"  Logs: {output_dir / 'logs'}")
            
            # List output files
            cutout_dir = output_dir / 'cutouts'
            bbox_dir = output_dir / 'bboxes'
            
            if cutout_dir.exists():
                cutout_files = sorted(cutout_dir.glob('*.png'))
                print(f"\n  Generated {len(cutout_files)} cutout images:")
                for f in cutout_files:
                    print(f"    - {f.name}")
            
            if bbox_dir.exists():
                bbox_files = sorted(bbox_dir.glob('*.png'))
                print(f"\n  Generated {len(bbox_files)} bounding box images:")
                for f in bbox_files:
                    print(f"    - {f.name}")
            
            print("\n" + "=" * 70)
            print("\nExample completed successfully!")
            print("\nTo run on your own data:")
            print("  python dinov2_tokencut_pipeline.py \\")
            print("    --input_dir /path/to/your/images \\")
            print("    --output_dir /path/to/output")
            print("\n" + "=" * 70)
            
            return 0
            
        except Exception as e:
            print(f"\n✗ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    sys.exit(run_example_pipeline())
