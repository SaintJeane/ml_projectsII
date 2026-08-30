<div align="center">

# 📚 Local RAG from Scratch

A Retrieval-Augmented Generation pipeline built without any RAG framework — no LangChain, no LlamaIndex — implementing chunking, embedding, semantic search, and local LLM generation directly with PyTorch and HuggingFace primitives.

---

<!-- Badges -->

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Gemma](https://img.shields.io/badge/LLM-Gemma-4285F4?style=for-the-badge&logo=google)](https://huggingface.co/google/gemma-2b-it)
[![sentence--transformers](https://img.shields.io/badge/Embeddings-sentence--transformers-yellow?style=for-the-badge&logo=huggingface)](https://www.sbert.net/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/SaintJeane/ml_projectsII/tree/main/rag_systems/local_rag)

</div>

> **Note:** This project exists as a reference notebook, not a deployed demo. It's intentionally the "from scratch" companion to the framework-based RAG systems elsewhere in this portfolio — see [Local Multimodal RAG](../multimodal_rag) and [Agentic RAG System](https://github.com/SaintJeane/agentic_rag), both of which build on LangChain/LangGraph rather than implementing retrieval primitives directly.

## Table of Contents

- [📚 Local RAG from Scratch](#-local-rag-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Pipeline](#pipeline)
  - [Chunking Strategy](#chunking-strategy)
  - [Semantic Search — Built from Primitives](#semantic-search--built-from-primitives)
  - [Hardware-Adaptive Model Selection](#hardware-adaptive-model-selection)
  - [Prompt Engineering](#prompt-engineering)
  - [Tech Stack](#tech-stack)
  - [Repository Structure](#repository-structure)
  - [Getting Started](#getting-started)
  - [Engineering Decisions \& Lessons Learned](#engineering-decisions--lessons-learned)
  - [Known Limitations](#known-limitations)
  - [Possible Extensions](#possible-extensions)

---

## Overview

Most RAG tutorials reach for LangChain or LlamaIndex immediately, which hides how retrieval actually works underneath. This notebook does the opposite: it builds every stage — sentence splitting, chunking, embedding, similarity search, prompt augmentation, and generation — directly with PyTorch, `sentence-transformers`, and `transformers`, to understand the mechanics before abstracting them away.

## Pipeline

```
PDF
 │
 ▼
Text extraction (PyMuPDF) + EDA on raw text stats
 │
 ▼
Sentence splitting (spaCy) → group into chunks of ~10 sentences per chunk
 │
 ▼
Filter out chunks too short to be useful
 │
 ▼
Embed chunks (sentence-transformers/all-mpnet-base-v2) → save to CSV
 │
 ▼
Query → embed query (same model) → exhaustive similarity search → top-k context
 │
 ▼
Prompt augmentation (few-shot examples + retrieved context) → Gemma generates the answer
```

## Chunking Strategy

Sentences are grouped into chunks of **10 sentences each** — a deliberate middle ground: large enough that chunks carry coherent context, small enough to fit comfortably inside the embedding model's context window and to stay easy to manually inspect and filter. On the reference PDF used in the notebook, this produced roughly **1.5 chunks per page on average**, consistent with most pages containing around 10 sentences.

## Semantic Search — Built from Primitives

Rather than reaching for a vector database, similarity search is implemented directly as a dot-product (or cosine similarity) comparison between the query embedding and every stored chunk embedding — an **exhaustive search**, not an approximate one.

The notebook explicitly reasons through *when* this is the right choice:
- **Dot product** emphasizes vector magnitude — appropriate when embeddings aren't normalized
- **Cosine similarity** emphasizes vector direction only — the more common choice for semantic text search
- Exhaustive search only stays practical up to a certain scale; the notebook explicitly calls out FAISS as the right tool once a dataset grows beyond what a single in-memory comparison can handle efficiently

At the scale used here — **1,680 chunk embeddings** — exhaustive dot-product search over every chunk completed in **0.00008 seconds**, meaning retrieval latency was a complete non-issue at this dataset size. This is a useful reference point for judging *when* a dedicated vector index actually becomes necessary versus when it's premature infrastructure.

## Hardware-Adaptive Model Selection

Rather than hardcoding which Gemma checkpoint to load, the notebook checks available GPU memory at runtime and picks a model size and quantization setting accordingly:

```python
if gpu_memory_gb < 5.1:
    # Not enough memory to run Gemma locally without quantization
elif gpu_memory_gb < 8.1:
    model_id = "google/gemma-2b-it"; use_quantization_config = True
elif gpu_memory_gb < 19.0:
    model_id = "google/gemma-2b-it"; use_quantization_config = False
else:  # > 19.0 GB
    model_id = "google/gemma-7b-it"; use_quantization_config = False
```

On the reference run (14.7GB available), this resolved to `gemma-2b-it` in float16, no quantization needed. The notebook also checks for Flash Attention 2 availability and GPU compute capability, falling back to standard scaled dot-product attention (`sdpa`) when Flash Attention isn't supported — so the same notebook runs correctly across a range of GPU tiers without manual editing.

## Prompt Engineering

The generation prompt isn't just "context + question." It follows a deliberate structure:

1. An explicit instruction to extract relevant passages internally before answering, but only return the final answer (a lightweight "think, then answer" pattern)
2. **Three worked examples** of the desired answer style and explanatory depth, baked directly into the prompt template as few-shot demonstrations
3. The retrieved context chunks, formatted as a bulleted list
4. The user's actual query

This few-shot structure is what pushes a small 2B-parameter instruction-tuned model toward consistent, explanatory answers rather than terse or inconsistent ones.

## Tech Stack

| Component | Technology |
|---|---|
| PDF parsing | PyMuPDF (`fitz`) |
| Sentence splitting | spaCy |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` |
| Similarity search | Exhaustive dot product / cosine similarity (PyTorch, no vector DB) |
| LLM | Google Gemma (2B or 7B, selected by available GPU memory), 4-bit quantization via `bitsandbytes` |
| Attention | Flash Attention 2 where supported, `sdpa` fallback otherwise |

## Repository Structure

```
local_rag/
├── Local_RAG_from_Scratch.ipynb   # Full pipeline: PDF → chunks → embeddings →
│                                    semantic search → prompt engineering → generation
└── README.md
```

## Getting Started

1. Open `Local_RAG_from_Scratch.ipynb` in Colab or a local Jupyter environment with a CUDA GPU (a GPU is recommended, not strictly required for the retrieval half)
2. Install: `torch`, `transformers`, `sentence-transformers`, `PyMuPDF`, `spacy`, `bitsandbytes`, `accelerate`
3. Place a PDF in the working directory and update the file path
4. Run all cells — the GPU-memory check will automatically select an appropriate Gemma checkpoint

```python
answer = ask(
    query="What are the macronutrients, and what roles do they play in the human body?",
    temperature=0.2,
    return_answer_only=True,
)
```

## Engineering Decisions & Lessons Learned

- **Built without a RAG framework, deliberately**: implementing chunking, embedding, and retrieval by hand rather than through LangChain/LlamaIndex trades convenience for a clear understanding of what those frameworks actually automate — a useful reference before adopting them elsewhere (as the other RAG projects in this portfolio do).
- **Exhaustive search over a vector index, at this scale**: with retrieval completing in well under a millisecond at 1,680 chunks, adding FAISS here would have been complexity without a corresponding benefit — the notebook explicitly reasons about *when* that tradeoff flips rather than defaulting to a vector DB by habit.
- **Runtime hardware detection over a fixed model choice**: checking GPU memory and Flash Attention support at runtime, rather than hardcoding a single model ID, means the same notebook is portable across different Colab GPU allocations without manual edits.
- **Few-shot examples embedded directly in the prompt template**: rather than relying on a larger, more expensive model to produce well-structured answers, three worked examples are baked into every prompt to steer a smaller 2B model's output style consistently.

## Known Limitations

- **No deployed demo**: this project exists as a local Colab/Jupyter notebook rather than a hosted Space. Gemma at 2B–7B parameters (even quantized) generally needs more sustained GPU access than HuggingFace's free CPU-tier Spaces provide — see the [Agentic RAG System](../../../agentic_rag/) in this portfolio for how a similar constraint was resolved via HuggingFace's ZeroGPU tier for a different model.
- **No formal retrieval or generation evaluation**: the notebook proposes using a second LLM to rate answer quality as a proxy evaluation metric, but doesn't implement it — this is flagged as a next step, not a finished evaluation framework.
- **Exhaustive search doesn't scale indefinitely**: fine at ~1,680 chunks; a larger document collection would need the FAISS-based approach the notebook explicitly calls out as the next step.

## Possible Extensions

- [ ] Advanced PDF extraction via [Marker](https://github.com/VikParuchuri/marker) instead of PyMuPDF alone
- [ ] Swap in alternative embedding models or LLMs (e.g. Mistral-Instruct)
- [ ] Vector database integration (FAISS) once document scale outgrows exhaustive search
- [ ] Streaming text output instead of a single blocking generation call
- [ ] An LLM-as-judge evaluation framework for answer quality
- [ ] Gradio interface for interactive querying
- [ ] NVIDIA TensorRT-LLM / Flash Attention 2 optimization for faster local generation

---