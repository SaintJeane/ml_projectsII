# Retrieval-Augmented Generation (RAG) systems — Overview

This repository contains example projects and notebooks that implement Retrieval-Augmented Generation (RAG) workflows. The focus is on simple, local and upload-ready multimodal RAG pipelines that combine PDF text extraction, image captioning, semantic search (FAISS) and LLM-based answer generation via a Gradio interface.

Contents (high-level)
---------------------

- `local_rag/` — Notebooks and experiments for building a RAG system from scratch locally. Contains a notebook `Local_RAG_from_Scratch.ipynb` with helper code for chunking, embeddings, FAISS indexing and query-time retrieval.
- `multimodal_rag/` — Notebooks and setup for a multimodal RAG system that handles images and text. Notable files:
	- `Simple_Local_Multimodal_RAG.ipynb` — self-contained notebook demonstrating the pipeline locally.
	- `MRAG_HF_Upload.ipynb` — prepares a `setup/` folder and helper scripts for uploading the RAG app to a Hugging Face Space (Gradio).

-----------------------


