# Multimodal RAG (MRAG) — README

This folder contains two Jupyter notebooks and a small setup for a local, simple multimodal Retrieval-Augmented Generation (RAG) system that:

- Extracts text and images from PDFs (via PyMuPDF / fitz).
- Generates captions/descriptions for images using a multimodal LLM (Gemma3).
- Chunks combined text + captions with LangChain text splitters.
- Builds semantic embeddings with SentenceTransformers and stores them in FAISS for retrieval.
- Streams answers from the LLM via a Gradio chat interface.

Notebooks
---------

- `MRAG_HF_Upload.ipynb` — code to prepare a small project directory (under `setup/multimodal_rag/`) that can be uploaded to a Hugging Face Space programmatically from the notebook. This notebook writes the following files into the `setup/multimodal_rag/` folder:
	- `app.py` — Gradio app that exposes the PDF upload + chat UI.
	- `main.py` — pipeline: extract pages, caption images, merge & chunk, create embeddings, FAISS index, semantic search, and answer generation/streaming.
	- `model_setup.py` — loads the embedding model and Gemma3 model + processor (example uses CPU/quantized options for Spaces compatibility).
	- `utils.py` — helper functions for cache/FAISS saving/loading, cleanup, and small utilities.
	- `requirements.txt` — packages to include when publishing to Hugging Face Space.
	- `README.md` (inside setup folder) — metadata block and short description used by Hugging Face Spaces.

- `Simple_Local_Multimodal_RAG.ipynb` — a self-contained notebook that implements the same pipeline locally (in-notebook). It demonstrates:
	- Loading and quantizing Gemma3 with bitsandbytes (optional).
	- Text and image extraction from PDFs.
	- Caption generation for images.
	- Chunking + creating embeddings, FAISS index management.
	- A small Gradio demo (in-notebook) to upload a PDF, build the index, and stream responses.

Quick features
--------------

- Multimodal captioning with `google/gemma-3-4b-it` (Gemma3) for image descriptions.
- Embeddings via `sentence-transformers/all-MiniLM-L6-v2` for semantic search.
- FAISS (IndexFlatIP) for fast similarity search.
- Gradio chat interface that streams responses using a TextIteratorStreamer style.

Requirements
------------

Recommended Python packages (see `setup/multimodal_rag/requirements.txt` created by the upload notebook):

- torch
- transformers
- sentence-transformers
- bitsandbytes
- accelerate
- numpy
- pillow
- PyMuPDF
- faiss-cpu
- langchain
- gradio

Notes on hardware
-----------------
- Gemma3 is large and ideally needs GPU. The notebooks include options for 4-bit quantization and device_map adjustments (e.g. CPU-only or device_map="auto"). If you plan to run locally with GPU, make sure CUDA and compatible PyTorch are installed. For deploying to the free tier of Hugging Face Spaces, the system targets CPU/quantized settings.

How to use (local notebook)
----------------------------

1. Open `Simple_Local_Multimodal_RAG.ipynb` and run the cells in order. Key workflow:
	 - Install requirements (if not already installed).
	 - Load/quantize model and processor, or skip/replace if you don't have access to Gemma3.
	 - Upload or place a PDF next to the notebook and set `pdf_path` and `image_dir` variables.
	 - Run preprocessing to extract text and images, generate captions, chunk text, build embeddings and a FAISS index.
	 - Run the Gradio demo cells to start a local interface for Q&A.

2. Example of calling the core `ask` helper (from the notebook):

```python
# from the notebook imports
answer = ask(
		query="Which country is Maasai Mara located in?",
		pdf_path="Maasai_Mara.pdf",
		image_dir="extracted_images",
		embedding_model=embedding_model,
		model=model,
		processor=processor,
		top_k=10
)
print(answer)
```

How to upload and run as a Hugging Face Space (from `MRAG_HF_Upload.ipynb`)
---------------------------------------------------------------------

1. Run `MRAG_HF_Upload.ipynb` to generate the `setup/multimodal_rag/` folder.
2. The notebook contains a programmatic upload sequence using `huggingface_hub.create_repo()` and `upload_folder()` to push the `setup/multimodal_rag/` folder to your HF Space (you need to be authenticated / set HF token in the environment).
3. The uploaded Space expects `app.py` as entrypoint and the `requirements.txt` included; the `README.md` (written into the setup folder) contains the top YAML metadata block required by Spaces.

Security & tokens
-----------------
- The notebooks show patterns for using a Hugging Face token (HF_TOKEN) from an environment or notebook secret. Do not hardcode tokens. For Colab you can use Google Colab's `userdata` or environment variables.

Limitations & suggestions
-------------------------
- Gemma3 is large and may require quantization or a powerful GPU for decent latency.
- The captioning step may produce noisy captions; trimming and cleaning (already included) helps improve retrieval relevance.
- Consider alternative lightweight image captioners if you can't run Gemma3 locally.
- Add persistence (S3 / remote storage) to share indexed FAISS stores between sessions for production scenarios.

Files created by the upload notebook
-----------------------------------

- setup/multimodal_rag/app.py     — Gradio app (entrypoint for Space)
- setup/multimodal_rag/main.py    — Core pipeline (extract, caption, chunk, embed, FAISS)
- setup/multimodal_rag/model_setup.py — Loads models and processors
- setup/multimodal_rag/utils.py   — helpers (save/load cache, FAISS helpers)
- setup/multimodal_rag/requirements.txt — requirements used by the Space
- setup/multimodal_rag/README.md — metadata and short description for HF Spaces

---

