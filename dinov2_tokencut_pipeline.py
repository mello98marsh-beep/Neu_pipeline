#!/usr/bin/env python3
"""
DINOv2 + TokenCut Object Localization Pipeline

This script implements an industry-standard pipeline for object localization using:
- DINOv2 as the feature extractor
- TokenCut for generating segmentation masks
- Connected component labeling for precise bounding box extraction

The pipeline processes images from <dataset>/<object> directories and saves:
- Segmented cutout images with '_sam_cutout.png' suffix
- Bounding box annotated images with '_sam_bboxes.png' suffix
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage
from PIL import Image

warnings.filterwarnings('ignore')


class DINOv2FeatureExtractor:
    """
    DINOv2 feature extractor module for generating image embeddings.
    """
    
    def __init__(self, model_name: str = 'dinov2_vits14', device: Optional[str] = None, 
                 test_mode: bool = False):
        """
        Initialize DINOv2 model.
        
        Args:
            model_name: Name of the DINOv2 model variant
            device: Device to run the model on (cuda/cpu)
            test_mode: If True, use a mock model for testing without downloading
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.test_mode = test_mode
        
        self.logger.info(f"Initializing DINOv2 model: {model_name} on {self.device}")
        
        if test_mode:
            self.logger.warning("Running in TEST MODE - using mock model")
            self._init_mock_model()
            return
        
        try:
            # Load DINOv2 model from torch hub
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get patch size from model
            self.patch_size = self.model.patch_size
            self.logger.info(f"DINOv2 model loaded successfully. Patch size: {self.patch_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to load DINOv2 model: {e}")
            self.logger.error("Please ensure you have internet access and/or the model is cached.")
            self.logger.error("You can pre-download models by running:")
            self.logger.error("  python -c \"import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')\"")
            self.logger.error("Alternatively, use --test_mode flag for testing without the model.")
            raise
    
    def _init_mock_model(self):
        """Initialize a mock model for testing purposes."""
        self.patch_size = 14
        self.model = None
        self.logger.info(f"Mock model initialized. Patch size: {self.patch_size}")
    
    def extract_features(self, image: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """
        Extract features from an image using DINOv2.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            features: Extracted features (N, D) where N is number of patches
            h_patches: Number of patches in height
            w_patches: Number of patches in width
        """
        h, w = image.shape[:2]
        
        # Resize image to be divisible by patch size
        new_h = (h // self.patch_size) * self.patch_size
        new_w = (w // self.patch_size) * self.patch_size
        
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h))
            self.logger.debug(f"Resized image from ({h}, {w}) to ({new_h}, {new_w})")
        
        # Calculate patch dimensions
        h_patches = new_h // self.patch_size
        w_patches = new_w // self.patch_size
        
        # If in test mode, return random features
        if self.test_mode:
            n_patches = h_patches * w_patches
            feature_dim = 384  # Standard DINOv2 small dimension
            features = torch.randn(n_patches, feature_dim)
            return features, h_patches, w_patches
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        image_tensor = image_tensor.to(self.device)
        
        # Normalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        image_tensor = (image_tensor - mean) / std
        
        with torch.no_grad():
            # Extract features
            features = self.model.forward_features(image_tensor)
            features = features['x_norm_patchtokens']  # (1, N, D)
        
        return features.squeeze(0), h_patches, w_patches


class TokenCutSegmentation:
    """
    TokenCut segmentation module for generating masks from features.
    """
    
    def __init__(self, tau: float = 0.2, eps: float = 1e-5, n_iter: int = 10):
        """
        Initialize TokenCut module.
        
        Args:
            tau: Threshold for binarization
            eps: Small epsilon for numerical stability
            n_iter: Number of power iteration steps
        """
        self.tau = tau
        self.eps = eps
        self.n_iter = n_iter
        self.logger = logging.getLogger(__name__)
    
    def compute_affinity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute affinity matrix from features.
        
        Args:
            features: Feature tensor (N, D)
            
        Returns:
            Affinity matrix (N, N)
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise similarities
        affinity = torch.mm(features, features.t())
        
        return affinity
    
    def power_iteration(self, affinity: torch.Tensor) -> torch.Tensor:
        """
        Apply power iteration to find eigenvector.
        
        Args:
            affinity: Affinity matrix (N, N)
            
        Returns:
            Eigenvector (N,)
        """
        N = affinity.shape[0]
        
        # Initialize with random vector
        v = torch.randn(N, 1, device=affinity.device)
        v = v / torch.norm(v)
        
        # Power iteration
        for _ in range(self.n_iter):
            v = torch.mm(affinity, v)
            v = v / (torch.norm(v) + self.eps)
        
        return v.squeeze()
    
    def generate_mask(self, features: torch.Tensor, h_patches: int, w_patches: int) -> np.ndarray:
        """
        Generate segmentation mask using TokenCut.
        
        Args:
            features: Feature tensor (N, D)
            h_patches: Number of patches in height
            w_patches: Number of patches in width
            
        Returns:
            Binary mask as numpy array (H, W)
        """
        self.logger.debug(f"Generating mask for {h_patches}x{w_patches} patches")
        
        # Compute affinity matrix
        affinity = self.compute_affinity_matrix(features)
        
        # Apply power iteration
        eigenvector = self.power_iteration(affinity)
        
        # Reshape to spatial dimensions
        eigenvector = eigenvector.reshape(h_patches, w_patches)
        
        # Normalize to [0, 1]
        eigenvector = (eigenvector - eigenvector.min()) / (eigenvector.max() - eigenvector.min() + self.eps)
        
        # Binarize using threshold
        mask = (eigenvector > self.tau).cpu().numpy().astype(np.uint8)
        
        # If mask is mostly empty, invert it
        if mask.sum() < mask.size * 0.1:
            mask = 1 - mask
            self.logger.debug("Mask was mostly empty, inverted it")
        
        return mask


class BoundingBoxExtractor:
    """
    Extract precise bounding boxes from segmentation masks using connected components.
    """
    
    def __init__(self, min_area: int = 100):
        """
        Initialize bounding box extractor.
        
        Args:
            min_area: Minimum area for a component to be considered
        """
        self.min_area = min_area
        self.logger = logging.getLogger(__name__)
    
    def extract_bboxes(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Extract bounding boxes from mask using connected component labeling.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            List of bounding boxes as (x_min, y_min, x_max, y_max)
        """
        # Apply morphological operations to clean the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find connected components
        labeled_mask, num_components = ndimage.label(mask)
        
        bboxes = []
        
        for label in range(1, num_components + 1):
            component_mask = (labeled_mask == label).astype(np.uint8)
            area = component_mask.sum()
            
            if area < self.min_area:
                continue
            
            # Find contours
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get bounding box from contour
                x, y, w, h = cv2.boundingRect(contours[0])
                bboxes.append((x, y, x + w, y + h))
        
        self.logger.debug(f"Extracted {len(bboxes)} bounding boxes")
        
        return bboxes


class PipelineOutput:
    """
    Handle saving of pipeline outputs.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize output handler.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.cutout_dir = self.output_dir / 'cutouts'
        self.bbox_dir = self.output_dir / 'bboxes'
        
        self.cutout_dir.mkdir(parents=True, exist_ok=True)
        self.bbox_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directories created: {self.output_dir}")
    
    def save_cutout(self, image: np.ndarray, mask: np.ndarray, filename: str):
        """
        Save segmented cutout image.
        
        Args:
            image: Original image (H, W, C)
            mask: Binary mask (H, W)
            filename: Original filename
        """
        # Resize mask to match image dimensions
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply mask to image
        cutout = image.copy()
        cutout[mask_resized == 0] = 255  # White background
        
        # Save with _sam_cutout.png suffix
        base_name = Path(filename).stem
        output_path = self.cutout_dir / f"{base_name}_sam_cutout.png"
        
        cv2.imwrite(str(output_path), cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))
        self.logger.info(f"Saved cutout: {output_path}")
    
    def save_bboxes(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]], 
                    mask: np.ndarray, filename: str):
        """
        Save image with bounding boxes annotated.
        
        Args:
            image: Original image (H, W, C)
            bboxes: List of bounding boxes (x_min, y_min, x_max, y_max) in mask coordinates
            mask: Binary mask (H, W) in patch resolution
            filename: Original filename
        """
        annotated = image.copy()
        
        # Calculate scaling factors
        scale_x = image.shape[1] / mask.shape[1]
        scale_y = image.shape[0] / mask.shape[0]
        
        # Draw bounding boxes
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            
            # Scale bbox to image coordinates
            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            x_max = int(x_max * scale_x)
            y_max = int(y_max * scale_y)
            
            # Draw rectangle
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        # Save with _sam_bboxes.png suffix
        base_name = Path(filename).stem
        output_path = self.bbox_dir / f"{base_name}_sam_bboxes.png"
        
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        self.logger.info(f"Saved bounding boxes: {output_path}")


class ObjectLocalizationPipeline:
    """
    Main pipeline orchestrator for DINOv2 + TokenCut object localization.
    """
    
    def __init__(self, args):
        """
        Initialize pipeline with configuration.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing Object Localization Pipeline")
        
        # Initialize modules
        try:
            self.feature_extractor = DINOv2FeatureExtractor(
                model_name=args.model_name,
                device=args.device,
                test_mode=args.test_mode
            )
            
            self.tokencut = TokenCutSegmentation(
                tau=args.tau,
                eps=args.eps,
                n_iter=args.n_iter
            )
            
            self.bbox_extractor = BoundingBoxExtractor(
                min_area=args.min_area
            )
            
            self.output_handler = PipelineOutput(args.output_dir)
            
            self.logger.info("Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.args.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / 'pipeline.log'
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.args.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Log after configuration is complete
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
    
    def get_image_paths(self) -> List[Path]:
        """
        Get all image paths from input directory.
        
        Returns:
            List of image paths
        """
        input_path = Path(self.args.input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_path.rglob(f'*{ext}'))
            image_paths.extend(input_path.rglob(f'*{ext.upper()}'))
        
        self.logger.info(f"Found {len(image_paths)} images in {input_path}")
        
        return sorted(image_paths)
    
    def process_image(self, image_path: Path):
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to the image file
        """
        try:
            self.logger.info(f"Processing: {image_path.name}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract features
            features, h_patches, w_patches = self.feature_extractor.extract_features(image)
            
            # Generate segmentation mask
            mask = self.tokencut.generate_mask(features, h_patches, w_patches)
            
            # Extract bounding boxes
            bboxes = self.bbox_extractor.extract_bboxes(mask)
            
            if not bboxes:
                self.logger.warning(f"No objects detected in {image_path.name}")
                # Still save outputs even if no bboxes detected
                bboxes = []
            
            # Save outputs
            self.output_handler.save_cutout(image, mask, image_path.name)
            self.output_handler.save_bboxes(image, bboxes, mask, image_path.name)
            
            self.logger.info(f"Successfully processed: {image_path.name}")
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
            if not self.args.continue_on_error:
                raise
    
    def run(self):
        """Execute the pipeline on all images."""
        try:
            image_paths = self.get_image_paths()
            
            if not image_paths:
                self.logger.warning("No images found to process")
                return
            
            self.logger.info(f"Starting pipeline processing for {len(image_paths)} images")
            
            for i, image_path in enumerate(image_paths, 1):
                self.logger.info(f"Progress: {i}/{len(image_paths)}")
                self.process_image(image_path)
            
            self.logger.info("Pipeline execution completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='DINOv2 + TokenCut Object Localization Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing images (supports <dataset>/<object> structure)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default='dinov2_vits14',
        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
        help='DINOv2 model variant'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (cuda/cpu). Auto-detect if not specified.'
    )
    
    # TokenCut arguments
    parser.add_argument(
        '--tau',
        type=float,
        default=0.2,
        help='Threshold for TokenCut binarization'
    )
    
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='Epsilon for numerical stability'
    )
    
    parser.add_argument(
        '--n_iter',
        type=int,
        default=10,
        help='Number of power iteration steps'
    )
    
    # Bounding box arguments
    parser.add_argument(
        '--min_area',
        type=int,
        default=100,
        help='Minimum area for detected components'
    )
    
    # Processing arguments
    parser.add_argument(
        '--test_mode',
        action='store_true',
        help='Run in test mode with mock DINOv2 model (for testing without model download)'
    )
    
    parser.add_argument(
        '--continue_on_error',
        action='store_true',
        help='Continue processing even if an image fails'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        pipeline = ObjectLocalizationPipeline(args)
        pipeline.run()
        
    except KeyboardInterrupt:
        logging.info("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
