<div align="center">

# Machine Learning Projects II

</div>

<!-- Core Languages & Frameworks -->
<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-f28500?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-d43f3a?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-Computer%20Vision-d43f3a?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-ffcc00?style=flat-square)](https://huggingface.co/docs/transformers/index)
[![LangChain](https://img.shields.io/badge/LangChain-Python-0052cc?style=flat-square)](https://www.langchain.com/)

</div>

<!-- Tools & Libraries -->
<div align="center">

[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-009688?style=flat-square)](https://github.com/facebookresearch/faiss)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-00bcd4?style=flat-square)](https://spacy.io/)
[![YOLO](https://img.shields.io/badge/YOLO-Detection-00457C?style=flat-square)](https://pjreddie.com/darknet/yolo/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-ff6f00?style=flat-square&logo=gradio)](https://www.gradio.app/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Applications-3f51b5?style=flat-square)](https://en.wikipedia.org/wiki/Computer_vision)
[![NLP RAG](https://img.shields.io/badge/NLP-RAG%20Systems-4caf50?style=flat-square)](https://huggingface.co/docs/transformers/main/en/rag)

</div>

<!-- Concepts -->
<div align="center">

[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-End--to--End-8bc34a?style=flat-square)](https://en.wikipedia.org/wiki/Machine_learning)
[![License](https://img.shields.io/badge/License-MIT-9e9e9e.svg?style=flat-square)](./LICENSE)

</div>

## Overview

Repository containing end-to-end machine learning projects on object detection (computer vision) and local retrieval-augmented generation (RAG) systems, including their evaluation and deployment.

## Projects

### 🗑️ [Trash Object Detection System](object_detection_CV/)
RT-DETRv2 fine-tuned to detect trash, hands, and bins in real-world photos, with three deliberate hard-negative classes (`not_trash`, `not_hand`, `not_bin`) so the model learns to reject near-misses rather than just recognize positives. Deployed as a gamified Gradio demo.
- **Live demo:** [Trash Object Detection](https://huggingface.co/spaces/Saint5/trash_object_detection_demo)
- **Result:** mAP@50 of 0.524 on a held-out test split; weaker on medium-sized objects (mAP 0.13), a known limitation tied to dataset scale.

### 📚 [Local RAG from Scratch](rag_systems/local_rag/)
A RAG pipeline built without any retrieval framework — chunking, embedding, semantic search, and generation implemented directly with PyTorch and HuggingFace primitives, using Gemma as the local LLM. A reference notebook, not a deployed demo — the deliberate "from scratch" companion to the framework-based systems below.

### 📄 [Local Multimodal RAG](rag_systems/multimodal_rag/)
Answers questions about uploaded PDFs by retrieving from both the document's text and captioned descriptions of its images, using a 4-bit quantized `google/gemma-3-4b-it` for both captioning and grounded answer generation, backed by a FAISS index.
- **Live demo:** [Multimodal RAG System](https://huggingface.co/spaces/Saint5/multimodal_rag_system)

Each project folder has its own README covering pipeline design, model/training details, evaluation results, engineering decisions, and known limitations.

## Repository Structure

```text
├── object_detection_CV/
│   ├── Drawing_Bounding_Box.ipynb
│   ├── Object_Detection_Notebook.ipynb
│   └── README.md
├── rag_systems/
│   ├── local_rag/
│   │   ├── Local_RAG_from_Scratch.ipynb
│   │   └── README.md
│   ├── multimodal_rag/
│   │   ├── Local_Multimodal_RAG.ipynb
│   │   ├── MRAG_HF_Upload.ipynb
│   │   └── README.md
│   └── README.md
├── README.md
├── .gitignore
└── LICENSE
```

## License

MIT — see [LICENSE](./LICENSE).