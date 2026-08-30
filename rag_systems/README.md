# Retrieval-Augmented Generation (RAG) Systems

This subfolder contains projects and notebooks that implement Retrieval-Augmented Generation (RAG) workflows, spanning both a from-scratch reference implementation and a deployed multimodal system.

## Contents

- **[`local_rag/`](local_rag/)** — [Local RAG from Scratch](local_rag/README.md): a RAG pipeline built without any retrieval framework — chunking, embedding, semantic search, and generation implemented directly with PyTorch and HuggingFace primitives. Notebook: [`Local_RAG_from_Scratch.ipynb`](local_rag/Local_RAG_from_Scratch.ipynb). Reference notebook only, not deployed.

- **[`multimodal_rag/`](multimodal_rag/)** — [Local Multimodal RAG](multimodal_rag/README.md): a RAG system that answers questions about uploaded PDFs by retrieving from both document text and captioned images, deployed live on Hugging Face Spaces. Notebooks:
  - [`Local_Multimodal_RAG.ipynb`](multimodal_rag/Local_Multimodal_RAG.ipynb) — self-contained pipeline, runs locally
  - [`MRAG_HF_Upload.ipynb`](multimodal_rag/MRAG_HF_Upload.ipynb) — packages and uploads the app to a Hugging Face Space

  👉 **[Live Demo](https://huggingface.co/spaces/Saint5/multimodal_rag_system)**

Each project's own README has full details: pipeline design, model and retrieval configuration, tech stack, engineering decisions, and known limitations.