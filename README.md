# DINOv2 + TokenCut Object Localization Pipeline

An industry-standard, production-ready pipeline for object localization using DINOv2 as a feature extractor and TokenCut for segmentation.

## Features

- 🎯 **Precise Object Localization**: Uses DINOv2 features with TokenCut segmentation
- 📦 **Modular Design**: Clean separation of concerns with reusable components
- 🔍 **Connected Component Analysis**: Accurate bounding box extraction
- 📊 **Comprehensive Logging**: Track pipeline execution with detailed logs
- ⚙️ **Configurable**: Extensive command-line arguments for customization
- 🛡️ **Robust**: Exception handling and error recovery

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mello98marsh-beep/Neu_pipeline.git
cd Neu_pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: For CUDA support, install PyTorch with the appropriate CUDA version:
```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage

Process images from a directory:

```bash
python dinov2_tokencut_pipeline.py \
    --input_dir /path/to/dataset/object \
    --output_dir /path/to/output
```

### Directory Structure

**Input**: The pipeline expects images in a directory structure like:
```
dataset/
  └── object/
      ├── image1.jpg
      ├── image2.png
      └── ...
```

**Output**: Results are saved in the following structure:
```
output/
  ├── cutouts/
  │   ├── image1_sam_cutout.png
  │   ├── image2_sam_cutout.png
  │   └── ...
  ├── bboxes/
  │   ├── image1_sam_bboxes.png
  │   ├── image2_sam_bboxes.png
  │   └── ...
  └── logs/
      └── pipeline.log
```

### Advanced Configuration

Full example with all options:

```bash
python dinov2_tokencut_pipeline.py \
    --input_dir ./dataset/cars \
    --output_dir ./results \
    --model_name dinov2_vitb14 \
    --device cuda \
    --tau 0.2 \
    --n_iter 10 \
    --min_area 100 \
    --continue_on_error \
    --log_level DEBUG
```

## Configuration Options

### Required Arguments

- `--input_dir`: Directory containing input images
- `--output_dir`: Directory to save results

### Model Configuration

- `--model_name`: DINOv2 model variant (default: `dinov2_vits14`)
  - Options: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`
  - Larger models provide better features but are slower
  
- `--device`: Computation device (default: auto-detect)
  - Options: `cuda`, `cpu`

### TokenCut Parameters

- `--tau`: Threshold for mask binarization (default: `0.2`)
  - Lower values: more permissive (larger masks)
  - Higher values: more restrictive (smaller masks)
  
- `--n_iter`: Power iteration steps (default: `10`)
  - More iterations: better convergence but slower
  
- `--eps`: Numerical stability epsilon (default: `1e-5`)

### Bounding Box Extraction

- `--min_area`: Minimum component area in pixels (default: `100`)
  - Filters out small noise components

### Processing Options

- `--continue_on_error`: Continue processing if an image fails
- `--log_level`: Logging verbosity (default: `INFO`)
  - Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## Output Files

### Cutout Images (`*_sam_cutout.png`)

Segmented objects with white background. The mask is applied to isolate the detected object.

### Bounding Box Images (`*_sam_bboxes.png`)

Original images with green bounding boxes drawn around detected objects. Multiple boxes may appear if multiple components are detected.

### Log Files (`logs/pipeline.log`)

Detailed execution logs including:
- Model initialization
- Feature extraction progress
- Segmentation results
- Errors and warnings

## Pipeline Architecture

The pipeline consists of five modular components:

1. **DINOv2FeatureExtractor**: Extracts dense features from images
2. **TokenCutSegmentation**: Generates segmentation masks using normalized cut
3. **BoundingBoxExtractor**: Derives tight bounding boxes with connected components
4. **PipelineOutput**: Handles saving of results
5. **ObjectLocalizationPipeline**: Orchestrates the entire workflow

## Technical Details

### DINOv2 Feature Extraction

- Uses Facebook's DINOv2 self-supervised vision transformer
- Extracts patch-level features with configurable model size
- Automatic image resizing to match patch dimensions

### TokenCut Segmentation

- Computes affinity matrix from normalized features
- Applies power iteration to find dominant eigenvector
- Binarizes using threshold to create foreground mask

### Bounding Box Extraction

- Applies morphological operations for noise reduction
- Uses connected component labeling for multi-object handling
- Filters components by minimum area threshold

## Performance Tips

1. **GPU Usage**: Use `--device cuda` for 10-50x speedup
2. **Model Selection**: Start with `dinov2_vits14` for speed, upgrade to larger models if needed
3. **Batch Processing**: The pipeline processes images sequentially; for large datasets, consider running multiple instances on different subsets
4. **Threshold Tuning**: Adjust `--tau` based on your dataset (try 0.1-0.3 range)

## Troubleshooting

### Out of Memory Errors

- Use a smaller model variant (`dinov2_vits14`)
- Process images sequentially (default behavior)
- Reduce input image resolution

### Poor Segmentation Results

- Adjust `--tau` threshold (try different values)
- Increase `--n_iter` for better convergence
- Try a larger model variant

### No Objects Detected

- Lower `--tau` threshold
- Reduce `--min_area` to detect smaller objects
- Check input images are valid and clear

## Examples

### Process a single object category:
```bash
python dinov2_tokencut_pipeline.py \
    --input_dir ./data/cars \
    --output_dir ./results/cars
```

### High-quality processing with large model:
```bash
python dinov2_tokencut_pipeline.py \
    --input_dir ./data/objects \
    --output_dir ./results \
    --model_name dinov2_vitl14 \
    --tau 0.15 \
    --n_iter 20
```

### Debug mode with detailed logging:
```bash
python dinov2_tokencut_pipeline.py \
    --input_dir ./test_images \
    --output_dir ./debug_output \
    --log_level DEBUG \
    --continue_on_error
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@inproceedings{wang2022tokencut,
  title={Tokencut: Segmenting objects in images and videos with self-supervised transformer and normalized cut},
  author={Wang, Yangtao and Shen, Xi and Hu, Shell Xu and Yuan, Yuan and Crowley, James L and Vaufreydaz, Dominique},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

## License

This project is provided as-is for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For questions or issues, please open an issue on GitHub.
