# Local RAG (Retrieval-Augmented Generation) from Scratch

This notebook demonstrates how to build a local RAG (Retrieval-Augmented Generation) system from scratch. The system combines document retrieval with language model generation to provide accurate, context-aware responses to queries using a local PDF document as the knowledge source.

## Overview

The notebook implements a complete RAG pipeline with the following components:

1. **Document Processing and Text Extraction**
   - PDF document loading and text extraction using PyMuPDF
   - Text preprocessing and formatting
   - Sentence splitting using spaCy
   - Text chunking for efficient processing

2. **Embedding Creation**
   - Utilizes the `sentence-transformers` library with the `all-mpnet-base-v2` model
   - Converts text chunks into numerical representations (embeddings)
   - Implements similarity search functionality

3. **Local LLM Integration**
   - Uses Google's Gemma model for local text generation
   - Supports both 2B and 7B parameter versions (depending on your hardware capability!)
   - Implements efficient GPU usage with quantization options

4. **RAG Pipeline Implementation**
   - Retrieval: Finding relevant document chunks for a query
   - Augmentation: Enhancing prompts with retrieved context
   - Generation: Producing context-aware responses

## Requirements

- Python 3.x
- PyTorch
- Transformers
- sentence-transformers
- PyMuPDF (for PDF processing)
- spaCy
- CUDA-capable GPU (recommended)

## Features

### Text Processing
- Advanced sentence splitting using spaCy
- Configurable chunk sizes for text processing
- Statistical analysis of text chunks
- Filtering options for short text chunks

### Embedding System
- High-quality embeddings using `all-mpnet-base-v2`
- Efficient similarity search implementation
- Support for both dot product and cosine similarity

### LLM Integration
- Automatic GPU memory management
- Support for model quantization
- Configurable generation parameters
- Prompt engineering with examples

## Usage Example

```python
# Initialize and use the RAG system
query = "What are the macronutrients, and what roles do they play in the human body?"
answer = ask(query=query,
            temperature=0.2,
            return_answer_only=True)
```

## Performance Considerations

- GPU memory requirements vary based on model size
- Quantization options available for limited GPU memory
- Batch processing for efficient embedding generation
- Optimized similarity search implementation

## Possible Extensions

1. Text Extraction Improvements
   - Integration with Marker
   - Advanced PDF extraction techniques

2. Model Enhancements
   - Alternative embedding models
   - Different LLM options (e.g., Mistral-Instruct)
   - Custom prompt engineering

3. System Improvements
   - Vector database integration for larger datasets
   - Streaming text output
   - Evaluation framework using additional LLMs
   - Integration with frameworks like LangChain/LlamaIndex

4. Performance Optimizations
   - GPU optimization techniques
   - NVIDIA TensorRT-LLM integration
   - Flash Attention 2 support

5. UI/Application Development
   - Gradio interface implementation
   - Chatbot functionality
   - Interactive visualization components

## References

- [Hugging Face Documentation](https://huggingface.co/docs)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [LangChain Documentation](https://python.langchain.com/docs/how_to/#text-splitters)