# Retrieval-Augmented Generation (RAG) systems

This subfolder contains projects and notebooks that implement Retrieval-Augmented Generation (RAG) workflows. The focus is on simple, local and upload-ready multimodal RAG pipelines that combine PDF text extraction, image captioning, semantic search (FAISS) and LLM-based answer generation via a Gradio interface.

## Overview

Contents (high-level)
---------------------

- `local_rag` — Notebooks and experiments for building a RAG system from scratch locally. Contains a notebook [`Local RAG from Scratch`](local_rag/local_rag_from_scratch.ipynb) with helper code for chunking, embeddings, FAISS indexing and query-time retrieval.
- [`multimodal_rag`](./multimodal_rag/) — Notebooks and setup for a multimodal RAG system that handles images and text. Notable files:
	- [`Local Multimodal RAG`](./multimodal_rag/local_multimodal_rag.ipynb) — self-contained notebook demonstrating the pipeline locally.
	- [`Multimodal RAG upload`](./multimodal_rag/mrag_hf_upload.ipynb) — prepares a `setup/` folder and modular helper scripts for uploading the RAG app to a Hugging Face Space (using Gradio app framework).

-----------------------


