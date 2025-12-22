# Azure OpenAI RAG API (FastAPI)

This project implements a Retrieval-Augmented Generation (RAG) API using
FastAPI, Azure OpenAI, LangChain, and FAISS.

The API answers user questions strictly based on provided documents.

---

## Tech Stack
- Python
- FastAPI
- Azure OpenAI
- LangChain
- FAISS
- HuggingFace Embeddings

---

## How it works
1. Documents are loaded from the data folder
2. Text is split into chunks
3. Embeddings are generated locally
4. FAISS performs similarity search
5. Azure OpenAI generates answers using retrieved context

---

## API Endpoint

POST /ask

Request:
{
  "question": "What safety measures are mentioned?"
}

Response:
{
  "answer": "Answer generated from document context"
}

---

## Run locally

uvicorn app.main:app --reload

Swagger UI:
http://127.0.0.1:8000/docs

---

Author: Ilakiya
