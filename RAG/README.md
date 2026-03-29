# RAG Medical Chatbot

This module implements a Retrieval-Augmented Generation pipeline for Vietnamese medical QA.

## Features

- Multi-file data loading from medical guideline JSON files
- Chunk-level document construction with metadata
- FAISS indexing and local cache
- Hybrid retrieval:
  - Priority keyword match
  - Semantic fallback via vector similarity
- Gemini-based answer generation with source-aware prompt
- Optional doctor-answer evaluation flow

## Folder Layout

- `data/`: medical knowledge JSON files used for retrieval
- `faiss_cache/`: persisted FAISS index
- `src/`: source code
- `test/`: test and debug scripts

## Environment Setup

1. Create and activate Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
copy .env.example .env
```

Then set `GOOGLE_API_KEY` in `.env`.

## Run Pipeline

Build FAISS cache:

```bash
cd src
python build_faiss.py
```

Run chatbot:

```bash
python main.py
```

## Key Configuration

Main settings are in `src/config.py`:
- `EMBEDDING_MODEL`
- `LLM_MODEL`
- `K_RETRIEVE`
- `MEDCHAT_RAG_BASE_DIR` (optional env override)

By default, base paths are auto-detected from the repository layout.

## Test Scripts

From project root, examples:

```bash
python RAG/test/test_data_loader.py
python RAG/test/test_embeddings_single.py
python RAG/test/test_faiss_single.py
python RAG/test/test_hybrid.py
python RAG/test/test_rag_single.py
```

## Notes for Recruiter/Reviewer

- Retrieval is grounded on real guideline sources under `data/`.
- Answer output includes top retrieved source snippets.
- The module is designed as an end-to-end demo for production-style medical assistant workflows.
