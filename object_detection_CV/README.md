<div align="center">

# 🗑️ Trash Object Detection System

An object detection model fine-tuned to recognize **trash**, **hands**, and **bins** in real-world photos - deployed as a gamified Gradio demo that awards a point when all three appear in the same frame.

---

<!-- Badges -->

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![RT--DETRv2](https://img.shields.io/badge/Model-RT--DETRv2-orange?style=for-the-badge)](https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2)

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/Saint5/trash_object_detection_demo)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/SaintJeane/ml_projectsII/tree/main/object_detection_CV)

</div>

## Table of Contents

- [🗑️ Trash Object Detection System](#️-trash-object-detection-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset \& Label Design](#dataset--label-design)
  - [Model \& Training](#model--training)
  - [Evaluation Results](#evaluation-results)
  - [The Deployed Demo](#the-deployed-demo)
  - [Tech Stack](#tech-stack)
  - [Repository Structure](#repository-structure)
  - [Getting Started](#getting-started)
  - [Engineering Decisions \& Lessons Learned](#engineering-decisions--lessons-learned)
  - [Known Limitations](#known-limitations)
  - [Possible Next Steps](#possible-next-steps)

---

## Overview

This project fine-tunes **RT-DETRv2** (`PekingU/rtdetr_v2_r50vd`) to detect three real-world objects — trash, a hand, and a bin — in a single photo, and packages it as an interactive Gradio demo with a small gamification hook: submit a photo containing all three, and the app returns a point.

```
Upload photo
     │
     ▼
RT-DETRv2 (fine-tuned) → bounding boxes + class labels + confidence scores
     │
     ▼
Check: are {trash, hand, bin} all present in this image?
     │
     ├── YES → "+1 Point 🪙 Found: [...]"
     └── NO  → "Missing: [...]" (names exactly what's absent)
```

## Dataset & Label Design

Trained on [`mrdbourke/trashify_manual_labelled_images`](https://huggingface.co/datasets/mrdbourke/trashify_manual_labelled_images), a hand-labelled image dataset.

The label set is **7 classes, not 3**:

| Class | Purpose |
|---|---|
| `trash` | The object being disposed of |
| `hand` | The hand holding the trash |
| `bin` | The disposal target |
| `trash_arm` | The arm extending toward the bin |
| `not_trash` | Hard negative — visually similar to trash, isn't |
| `not_hand` | Hard negative — visually similar to a hand, isn't |
| `not_bin` | Hard negative — visually similar to a bin, isn't |

The three `not_*` classes are a deliberate design choice: rather than only ever training the model to say "yes, this is trash," the dataset also teaches it to actively reject near-misses. This is a stronger signal than a 3-class positive-only setup, at the cost of needing more careful manual labelling up front.

## Model & Training

- **Base model:** `PekingU/rtdetr_v2_r50vd`, loaded via `AutoModelForObjectDetection` / `AutoImageProcessor`
- **Fine-tuning:** HuggingFace `Trainer`, 10 epochs
- **Discriminative learning rates:** a custom `Trainer` subclass applies two different learning rates —

  ```python
  BACKBONE_LEARNING_RATE = 1e-5          # pretrained ResNet-50 backbone — small updates
  DETECTION_HEAD_LEARNING_RATE = 1e-4    # detection head — larger updates, learning from scratch
  ```

  This reflects a standard transfer-learning principle: the backbone already has useful pretrained features and should be nudged gently, while the detection head is being learned from scratch and needs bigger steps.

- **Other hyperparameters:**

  | Parameter | Value | Purpose |
  |---|---|---|
  | `BATCH_SIZE` | 8 (reduced from 16) | Avoid out-of-memory errors during training |
  | `WEIGHT_DECAY` | 1e-4 | Penalizes overly large weights over time |
  | `MAX_GRAD_NORM` | 0.1 | Clips gradients to prevent unstable updates |
  | `WARMUP_RATIO` | 0.05 | Learning rate ramps up over the first 5% of training steps |
  | `lr_scheduler_type` | `linear` | Linear decay after warmup |

## Evaluation Results

Measured on a held-out test split using `torchmetrics.detection.MeanAveragePrecision`:

| Metric | Score |
|---|---|
| mAP (overall) | **0.3806** |
| mAP@50 (IoU ≥ 0.50) | **0.5244** |
| mAP@75 (IoU ≥ 0.75) | **0.4268** |
| mAP — small objects | 0.3000 |
| mAP — medium objects | **0.1304** |
| mAP — large objects | 0.4038 |
| mAR@100 | 0.6919 |

The model performs noticeably worse on medium-sized objects than on small or large ones — see [Known Limitations](#known-limitations) for the likely cause.

## The Deployed Demo

The fine-tuned checkpoint is hosted on the HuggingFace Hub at `Saint5/rt_detrv2_finetuned_trash_box_detector_v1` and loaded directly by the Gradio app.

**Demo interface:**
- Image upload
- **Adjustable confidence threshold** slider (default `0.3`) — lets the user trade off precision vs. recall live, rather than baking in a single fixed cutoff
- Colour-coded bounding boxes drawn per class (green = bin, blue = trash, purple = hand, yellow = trash_arm, red = the `not_*` hard-negative classes)
- A text response that either confirms all three target items were found ("+1 Point 🪙") or names exactly which of `{trash, hand, bin}` is still missing from the frame

👉 **[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/Saint5/trash_object_detection_demo)**

## Tech Stack

| Component | Technology |
|---|---|
| Detection model | RT-DETRv2 (`PekingU/rtdetr_v2_r50vd`), fine-tuned |
| Framework | PyTorch + HuggingFace `transformers` |
| Training loop | HuggingFace `Trainer` (custom subclass for discriminative LR) |
| Evaluation | `torchmetrics.detection.MeanAveragePrecision` |
| Image handling | `PIL.ImageDraw`, `torchvision.ops.box_convert` |
| Deployment | Gradio `Interface`, HuggingFace Spaces |

## Repository Structure

```
object_detection_CV/
├── Drawing_Bounding_Box.ipynb       # Bounding box formats & conversions tutorial
│                                     #   XYXY (PyTorch) ↔ XYWH ↔ CXCYWH (YOLO)
│                                     #   drawing via PIL, Matplotlib, and Torchvision
│
├── Object_Detection_Notebook.ipynb  # Full detection pipeline, end to end:
│                                     #   data loading → COCO-format annotation prep →
│                                     #   model + processor setup → training → evaluation →
│                                     #   Gradio app + HuggingFace Spaces deployment
│
└── README.md
```

## Getting Started

1. Open `Object_Detection_Notebook.ipynb` in Google Colab or Jupyter
2. Install dependencies: `torch`, `torchvision`, `transformers`, `pillow`, `matplotlib`, `numpy` (Python 3.12)
3. Run all cells — the dataset downloads automatically from the HuggingFace Hub
4. The final cells package and upload a Gradio demo (`app.py`, `requirements.txt`, example images) to a HuggingFace Space

To try the bounding-box format conversions on their own, `Drawing_Bounding_Box.ipynb` is self-contained and doesn't require training a model.

## Engineering Decisions & Lessons Learned

- **Hard-negative classes over a simple 3-class setup**: adding `not_trash`, `not_hand`, and `not_bin` as their own labels — rather than just training on the three positive classes — pushes the model to actively discriminate near-misses instead of defaulting to "detect everything that looks plausible."

- **Discriminative learning rates via a custom `Trainer` subclass**: rather than applying one learning rate across the whole model, backbone and detection-head parameters are separated into two parameter groups with different rates (`1e-5` vs `1e-4`), respecting that the backbone is pretrained and the head is not.

- **Batch size reduced from 16 to 8**: the original batch size of 16 triggered out-of-memory errors during training; dropping to 8 resolved it at the cost of noisier gradient estimates per step.

- **Confidence threshold exposed to the end user, not fixed**: instead of hardcoding a single detection threshold, the deployed demo exposes it as a slider (default `0.3`), since the right threshold genuinely depends on the image and what the user is trying to detect.

## Known Limitations

- **Weaker performance on medium-sized objects** (mAP 0.1304 vs. 0.30–0.40 for small/large) — likely due to the hand-labelled dataset's limited size and the natural difficulty of scale-invariant detection with fewer medium-scale training examples in that range.
- **Hand-labelled dataset scale**: as a manually annotated dataset, its size is inherently smaller than dataset-scraped alternatives, which caps how much the model can generalize to unseen backgrounds and lighting.
- **No formal hyperparameter search**: the notebook trains for a fixed 10 epochs and explicitly raises the open question of whether longer training would improve results, without yet testing it.
- **Single confidence threshold applies globally**: the same threshold is applied across all 7 classes; classes with different natural score distributions might benefit from per-class thresholds.

## Possible Next Steps

- [ ] Test whether training beyond 10 epochs improves mAP, particularly for medium-sized objects
- [ ] Expand the hand-labelled dataset, particularly with more medium-scale object examples
- [ ] Evaluate per-class confidence thresholds instead of one global threshold
- [ ] Add a written benchmark comparing inference latency across CPU vs. GPU Spaces tiers

---