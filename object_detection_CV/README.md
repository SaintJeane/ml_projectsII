# Object Detection Project

This repository contains two notebooks focusing on object detection tasks:

## Drawing_Bounding_Box.ipynb
A comprehensive guide to working with bounding boxes in computer vision, covering:

- Different bounding box formats and their use cases:
  - XYXY format `[x1, y1, x2, y2]` (used in PyTorch)
  - XYWH format `[x, y, width, height]`
  - CXCYWH format `[center_x, center_y, width, height]` (used in YOLO)

- Converting between different bounding box formats:
  - Using `torchvision.ops.box_convert`
  - Manual conversion implementations

- Normalization of bounding box coordinates

- Drawing bounding boxes using different methods:
  - PIL.ImageDraw
  - Matplotlib
  - Torchvision

## Object_Detection_Notebook.ipynb
A complete implementation of an object detection model that detects trash, hands, and bins. Key features include:

- Model: Uses Hugging Face's `AutoImageProcessor` and `AutoModelForObjectDetection` with the `PekingU/rtdetr_v2_r50vd` model
- Dataset management and preprocessing
- Training pipeline implementation
- Visualization tools

### Features
- Data loading and preprocessing
- Model setup and configuration
- Training/validation/test split management
- Batch processing
- Model evaluation

## Requirements
- Python 3.x
- PyTorch
- torchvision
- transformers
- PIL
- matplotlib
- numpy

## Usage

### For Drawing_Bounding_Box.ipynb
```python
# Example for converting box formats
from torchvision.ops import box_convert

# Convert XYWH to XYXY format
box_xyxy = box_convert(boxes=torch.tensor(box_xywh), 
                      in_fmt='xywh', 
                      out_fmt='xyxy')
```

### For Object_Detection_Notebook.ipynb
```python
# Example for loading model and processor
from transformers import AutoModelForObjectDetection, AutoImageProcessor

MODEL_NAME = "PekingU/rtdetr_v2_r50vd"

# Load model and processor
model = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME,
    label2id=label2id,
    id2label=id2label
)

image_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    use_fast=True
)
```

## Project Structure
```
object_detection_CV/
├── Drawing_Bounding_Box.ipynb      # Bounding box tutorial
├── Object_Detection_Notebook.ipynb  # Object detection implementation
└── README.md                       # Project documentation
```
