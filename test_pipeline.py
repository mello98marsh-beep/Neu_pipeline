#!/usr/bin/env python3
"""
Test script for DINOv2 + TokenCut Pipeline

This script validates the pipeline structure and core functionality.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2


def create_test_image(size=(224, 224)):
    """Create a simple test image with a rectangle."""
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    # Draw a colored rectangle in the center
    h, w = size
    cv2.rectangle(image, (w//4, h//4), (3*w//4, 3*h//4), (255, 0, 0), -1)
    
    return image


def test_pipeline_structure():
    """Test that the pipeline script has correct structure."""
    print("Testing pipeline structure...")
    
    try:
        # Import the pipeline
        import dinov2_tokencut_pipeline as pipeline
        
        # Check required classes exist
        assert hasattr(pipeline, 'DINOv2FeatureExtractor')
        assert hasattr(pipeline, 'TokenCutSegmentation')
        assert hasattr(pipeline, 'BoundingBoxExtractor')
        assert hasattr(pipeline, 'PipelineOutput')
        assert hasattr(pipeline, 'ObjectLocalizationPipeline')
        
        print("✓ All required classes are present")
        
        # Check main function exists
        assert hasattr(pipeline, 'main')
        assert hasattr(pipeline, 'parse_arguments')
        
        print("✓ Main entry points are present")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline structure test failed: {e}")
        return False


def test_argument_parser():
    """Test that argument parser works correctly."""
    print("\nTesting argument parser...")
    
    try:
        import dinov2_tokencut_pipeline as pipeline
        
        # Create parser and parse with explicit arguments
        parser = pipeline.argparse.ArgumentParser()
        # Manually add arguments to avoid calling parse_arguments
        args = pipeline.parse_arguments.__wrapped__ if hasattr(pipeline.parse_arguments, '__wrapped__') else None
        
        # Alternative approach: parse with explicit args list
        test_args = [
            '--input_dir', '/tmp/test_input',
            '--output_dir', '/tmp/test_output'
        ]
        
        # Temporarily save and restore sys.argv
        original_argv = sys.argv.copy()
        try:
            sys.argv = ['test'] + test_args
            args = pipeline.parse_arguments()
        finally:
            sys.argv = original_argv
        
        assert args.input_dir == '/tmp/test_input'
        assert args.output_dir == '/tmp/test_output'
        assert args.model_name == 'dinov2_vits14'  # default
        assert args.tau == 0.2  # default
        
        print("✓ Argument parser works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Argument parser test failed: {e}")
        return False


def test_output_handler():
    """Test the output handler."""
    print("\nTesting output handler...")
    
    try:
        import dinov2_tokencut_pipeline as pipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create output handler
            output_handler = pipeline.PipelineOutput(tmpdir)
            
            # Check directories were created
            assert (Path(tmpdir) / 'cutouts').exists()
            assert (Path(tmpdir) / 'bboxes').exists()
            
            print("✓ Output directories created successfully")
            
            # Test saving a cutout
            test_image = create_test_image()
            test_mask = np.zeros((28, 28), dtype=np.uint8)  # Small mask
            test_mask[10:20, 10:20] = 1
            
            output_handler.save_cutout(test_image, test_mask, "test.jpg")
            
            cutout_file = Path(tmpdir) / 'cutouts' / 'test_sam_cutout.png'
            assert cutout_file.exists()
            
            print("✓ Cutout saving works correctly")
            
            # Test saving bboxes
            test_bboxes = [(10, 10, 20, 20)]
            output_handler.save_bboxes(test_image, test_bboxes, test_mask, "test.jpg")
            
            bbox_file = Path(tmpdir) / 'bboxes' / 'test_sam_bboxes.png'
            assert bbox_file.exists()
            
            print("✓ Bounding box saving works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Output handler test failed: {e}")
        return False


def test_bbox_extractor():
    """Test the bounding box extractor."""
    print("\nTesting bounding box extractor...")
    
    try:
        import dinov2_tokencut_pipeline as pipeline
        
        extractor = pipeline.BoundingBoxExtractor(min_area=10)
        
        # Create a simple mask with two components
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # Component 1
        mask[60:80, 60:80] = 1  # Component 2
        
        bboxes = extractor.extract_bboxes(mask)
        
        assert len(bboxes) > 0, "Should detect at least one component"
        
        # Check bbox format
        for bbox in bboxes:
            assert len(bbox) == 4, "Bbox should have 4 coordinates"
            x_min, y_min, x_max, y_max = bbox
            assert x_max > x_min, "x_max should be greater than x_min"
            assert y_max > y_min, "y_max should be greater than y_min"
        
        print(f"✓ Detected {len(bboxes)} bounding boxes correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Bounding box extractor test failed: {e}")
        return False


def test_tokencut_segmentation():
    """Test the TokenCut segmentation module."""
    print("\nTesting TokenCut segmentation...")
    
    try:
        import torch
        import dinov2_tokencut_pipeline as pipeline
        
        tokencut = pipeline.TokenCutSegmentation(tau=0.2)
        
        # Create dummy features
        n_patches = 14 * 14  # 14x14 grid
        feature_dim = 384
        features = torch.randn(n_patches, feature_dim)
        
        # Generate mask
        mask = tokencut.generate_mask(features, 14, 14)
        
        assert mask.shape == (14, 14), "Mask shape should match patch dimensions"
        assert mask.dtype == np.uint8, "Mask should be uint8"
        assert set(np.unique(mask)).issubset({0, 1}), "Mask should be binary"
        
        print("✓ TokenCut segmentation works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ TokenCut segmentation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DINOv2 + TokenCut Pipeline Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Pipeline Structure", test_pipeline_structure()))
    results.append(("Argument Parser", test_argument_parser()))
    results.append(("Output Handler", test_output_handler()))
    results.append(("Bounding Box Extractor", test_bbox_extractor()))
    results.append(("TokenCut Segmentation", test_tokencut_segmentation()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
