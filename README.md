# med-chat

Medical chatbot project combining:
- LLM fine-tuning for domain adaptation (folder `Finetune/`)
- Retrieval-Augmented Generation (RAG) for grounded answers (folder `RAG/`)

## Project Structure

- `Finetune/`
  - Medical instruction datasets and fine-tuning notebooks
  - Fine-tuned adapter artifacts in `OutputModel/`
  - Evaluation result files in `MetricResult/`
- `RAG/`
  - Medical guideline JSON data
  - FAISS cache for retrieval
  - Source code for loading, indexing, retrieval, and chatbot inference
  - Test scripts for loader, embeddings, FAISS, retriever, and single-query checks

## What This Repository Demonstrates

- End-to-end medical chatbot workflow: data preparation -> retrieval index -> answer generation
- Hybrid retrieval design (keyword priority + semantic fallback)
- Prompted generation with Gemini API and source-grounded responses
- Practical evaluation/test scripts for retrieval and response stability

## Quick Start

Use the RAG module first:
1. Open `RAG/README.md`
2. Install dependencies from `RAG/requirements.txt`
3. Configure environment variables from `RAG/.env.example`
4. Build FAISS index and run chatbot

## Notes

- `Finetune/` notebooks target GPU/Colab workflows and can require a different dependency stack than `RAG/`.
- Keep API keys in environment variables, never hardcode in source files.
