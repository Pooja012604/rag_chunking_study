# RAG Chunking Study (SQuAD v1.1)

This repository contains code and experiments for studying how fixed-size chunking affects retrieval groundedness in a RAG pipeline using SQuAD v1.1.

## Experiments

- Chunking strategies:
  - Small: 200 words, 50 overlap
  - Medium: 400 words, 80 overlap
  - Large: 800 words, 120 overlap
- Retriever: all-MiniLM-L6-v2 + FAISS (cosine similarity)
- Metrics:
  - Grounded@k (k = 1, 2, 4)
  - Optional end-to-end QA evaluation using Ollama (phi3:mini)

### Create venv (recommended)

````powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

## How to Run
```bash
pip install -r requirements.txt
python .\src\run_rag_chunking.py
python .\src\score_results.py
python .\src\make_plot.py
## Setup

## Paper (Zenodo)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18226194.svg)](https://doi.org/10.5281/zenodo.18226194)

````
