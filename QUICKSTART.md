# Quick Start Guide

## Installation
```bash
pip install -r requirements.txt
```

## Run Example
```bash
python example.py
```

## Basic Usage
```bash
python dinov2_tokencut_pipeline.py \
    --input_dir <path/to/dataset/object> \
    --output_dir <path/to/output>
```

## Key Features
✓ DINOv2 feature extraction  
✓ TokenCut segmentation  
✓ Connected component bounding boxes  
✓ Industry-standard error handling  
✓ Comprehensive logging  
✓ Configurable parameters  

## Output Structure
```
output/
  ├── cutouts/          # *_sam_cutout.png files
  ├── bboxes/           # *_sam_bboxes.png files
  └── logs/             # pipeline.log
```

## Common Parameters
- `--model_name`: dinov2_vits14 (default), dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
- `--tau`: Threshold for segmentation (default: 0.2)
- `--device`: cuda or cpu (auto-detect by default)
- `--test_mode`: Use mock model for testing

## Full Documentation
See [README.md](README.md) for complete documentation.
