<div align="center">

# 📄 Local Multimodal RAG (MRAG)

A Retrieval-Augmented Generation system that answers questions about uploaded PDFs by retrieving from **both the document's text and captioned descriptions of its images** — powered by a quantized multimodal LLM running on free-tier hardware.

---

<!-- Badges -->

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Gemma3](https://img.shields.io/badge/LLM-Gemma3--4B--it-4285F4?style=for-the-badge&logo=google)](https://huggingface.co/google/gemma-3-4b-it)
[![LangChain](https://img.shields.io/badge/LangChain-Chunking-green?style=for-the-badge)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?style=for-the-badge&logo=gradio)](https://gradio.app/)

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/Saint5/multimodal_rag_system)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/SaintJeane/ml_projectsII/tree/main/rag_systems/multimodal_rag)

</div>

## Table of Contents

- [📄 Local Multimodal RAG (MRAG)](#-local-multimodal-rag-mrag)
	- [Table of Contents](#table-of-contents)
	- [Overview](#overview)
	- [Pipeline](#pipeline)
	- [Model \& Quantization](#model--quantization)
	- [Retrieval Design](#retrieval-design)
	- [Answer Generation — Grounded, Not Open-Ended](#answer-generation--grounded-not-open-ended)
	- [Tech Stack](#tech-stack)
	- [Live Demo](#live-demo)
	- [Repository Structure](#repository-structure)
	- [Getting Started](#getting-started)
	- [Engineering Decisions \& Lessons Learned](#engineering-decisions--lessons-learned)
	- [Known Limitations](#known-limitations)
	- [Suggested Improvements](#suggested-improvements)

---

## Overview

A user uploads a PDF; the system extracts its text and images **separately**, generates natural-language captions for every image using a multimodal LLM, then merges captions back into the surrounding page text before chunking, embedding, and indexing everything together. This means a question like *"what does the chart on page 4 show?"* can be answered even though the chart itself is an image — its caption becomes retrievable text.

## Pipeline

```
PDF upload
    │
    ▼
┌─────────────────────────────┐
│  Per-page extraction         │  PyMuPDF (fitz): text + embedded images, per page
└──────────────┬────────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Image captioning             │  Gemma3 (multimodal) describes each image
│                               │  Images < 32×32px are filtered out first
└──────────────┬────────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Merge + chunk                │  Captions appended inline under each page's text
│                               │  RecursiveCharacterTextSplitter, 1000 chars / 200 overlap
└──────────────┬────────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Embed + index                 │  all-MiniLM-L6-v2 → FAISS IndexFlatIP
└──────────────┬────────────────┘
               │
               ▼
     Query → top-k semantic search → grounded-only answer generation → streamed to UI
```

## Model & Quantization

The captioning **and** answer-generation model is a single multimodal LLM, `google/gemma-3-4b-it`, loaded via `Gemma3ForConditionalGeneration` and quantized for free-tier compute:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.bfloat16,
)
```

4-bit NF4 quantization with double quantization keeps a 4B-parameter multimodal model within reach of CPU/free-GPU-tier HuggingFace Spaces — the same tradeoff pattern used elsewhere in this portfolio's [Agentic RAG project](../../../agentic_rag) to fit a 7B model onto ZeroGPU.

## Retrieval Design

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Index:** FAISS `IndexFlatIP` (inner-product search — effectively cosine similarity over normalized embeddings)
- **Chunking:** `RecursiveCharacterTextSplitter` with a separator cascade from paragraph → sentence → word (`["\n\n", "\n", ".", " ", ""]`), 1000-character chunks with 200-character overlap, so context isn't lost at chunk boundaries
- **Retrieval depth:** top-`k=10` chunks per query, passed as grounding context
- **Caching:** the FAISS index and chunk metadata are persisted to disk (`index.faiss`, `chunks.json`) and reloaded if present — so re-querying the same PDF skips the entire extraction/captioning/chunking pipeline

## Answer Generation — Grounded, Not Open-Ended

The system prompt explicitly constrains the model to the retrieved context only:

> *"Use only the following pieces of retrieved context to answer the question... If the answer is not found in the provided context, state that the information is not available in the document. Do not use any external knowledge or make assumptions."*

This is a deliberate faithfulness choice: the model can produce a fluent, plausible-sounding answer from its own training knowledge even when the PDF doesn't contain the relevant information, which is precisely the failure mode a document QA system needs to avoid. The prompt trades away some helpfulness (the model won't fill in gaps from general knowledge) for grounding guarantees.

## Tech Stack

| Component | Technology |
|---|---|
| PDF parsing | PyMuPDF (`fitz`) — text and embedded images, per page |
| Multimodal LLM | `google/gemma-3-4b-it` (captioning + answer generation), 4-bit NF4 quantized |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS `IndexFlatIP` |
| UI | Gradio `Blocks` + `ChatInterface`, streaming responses |

## Live Demo

👉 **[Multimodal RAG System on HuggingFace Spaces](https://huggingface.co/spaces/Saint5/multimodal_rag_system)**

Upload a PDF (≤50 pages recommended), click "Process PDF," then ask questions about it in the chat panel.

## Repository Structure

```
multimodal_rag/
├── Local_Multimodal_RAG.ipynb   # Self-contained pipeline: extraction → captioning →
│                                  chunking → embedding → FAISS → in-notebook Gradio demo
│
├── MRAG_HF_Upload.ipynb          # Packages the pipeline into a standalone HF Space:
│                                  #   app.py         — Gradio entrypoint
│                                  #   main.py        — core pipeline
│                                  #   model_setup.py — model/processor loading
│                                  #   utils.py       — caching, cleanup helpers
│                                  #   requirements.txt
│                                  # Then uploads via huggingface_hub.create_repo() + upload_folder()
│
└── README.md
```

## Getting Started

1. Open `Local_Multimodal_RAG.ipynb` in Colab or Jupyter (GPU recommended for Gemma3)
2. Install dependencies: `torch`, `transformers`, `sentence-transformers`, `bitsandbytes`, `accelerate`, `faiss-cpu`, `langchain`, `PyMuPDF`, `gradio`, `pillow`
3. Place a PDF next to the notebook and set `pdf_path` / `image_dir`
4. Run all cells — extraction, captioning, chunking, indexing, and the Gradio demo launch in sequence

Programmatic query example (from the notebook):

```python
answer = ask(
    query="Which country is Maasai Mara located in?",
    pdf_path="Maasai_Mara.pdf",
    image_dir="extracted_images",
    embedding_model=embedding_model,
    model=model,
    processor=processor,
    top_k=10,
)
```

To deploy your own copy as a HuggingFace Space, run `MRAG_HF_Upload.ipynb` with an `HF_TOKEN` set in your environment (never hardcoded — the notebook reads it from Colab secrets or an env var).

## Engineering Decisions & Lessons Learned

- **Captions merged as inline text, not stored separately**: image descriptions are appended directly under each page's extracted text (`[Image Description]: ...`) before chunking, rather than embedded and indexed as a separate modality. This keeps retrieval simple — one embedding space, one FAISS index — at the cost of losing any signal about *which* part of a chunk came from an image versus body text.

- **Small images filtered before captioning**: images under 32×32 pixels are skipped entirely. These are almost always decorative artifacts (icons, spacers, logos) rather than meaningful content, and captioning them wastes a full LLM forward pass for no retrieval value.

- **Extracted images deleted after captioning**: once captions are generated, the extracted image files are cleaned up from disk. The captions (as text) are what get indexed — the images themselves are a transient intermediate artifact, not something the free-tier Space needs to retain.

- **FAISS index cached to disk per document**: reprocessing the same PDF on a second query check first for an existing `index.faiss` / `chunks.json` pair and skips extraction, captioning, and embedding entirely if found — meaningful savings given that captioning is the most expensive step in the pipeline.

## Known Limitations

- **Gemma3 is compute-heavy**: even 4-bit quantized, a 4B multimodal model has real latency on CPU-only free-tier hardware — this is explicitly why the demo recommends PDFs of 50 pages or fewer.
- **Caption quality varies**: image captions can be noisy or generic depending on image complexity, which directly affects retrieval relevance for anything answerable only from an image.
- **No persistence across sessions in the deployed Space**: the FAISS cache is disk-based within a single Space instance; it doesn't survive a Space restart or scale across multiple users' documents simultaneously.
- **Single global top-k**: retrieval always pulls the same number of chunks (10) regardless of query complexity or document length.

## Suggested Improvements

These are noted directly in the reference notebook as next steps:

- [ ] Make the system agentic by integrating LangChain/LangGraph — allowing the agent to decide when to retrieve, re-query, or reason over multiple steps rather than a fixed retrieve-then-answer pipeline
- [ ] Enable live internet search as a fallback when the uploaded document doesn't contain enough information to answer confidently

---