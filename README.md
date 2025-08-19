# Intelligent-PDF-Summarizer-QA-Bot

🚀 Project Overview

This project is an AI-powered PDF Summarizer & QA Bot that can:

Extract text from PDF files (with OCR fallback for scanned pages).

Generate a structured, hierarchical summary of the document.

Answer user queries with page-number citations.

Provide sample queries for quick insights.

It is built with LangChain, Hugging Face embeddings, ChromaDB, and Ollama LLaMA3 model.

🛠️ Features

✅ PDF Text Extraction (PyMuPDF)

✅ OCR Fallback (pytesseract + pdf2image)

✅ Chunking & Embeddings (sentence-transformers/all-MiniLM-L6-v2)

✅ Vector Store using ChromaDB

✅ Structured Summary Generation (LLaMA3)

✅ Retrieval-Augmented QA with page citations

✅ Streamlit UI for easy interaction

✅ Sample Queries for testing

📂 Project Flow

Upload a PDF file.

Extract text (OCR is used if text is missing).

Split text into chunks and generate embeddings.

Store embeddings in Chroma vector database.

Use LLaMA3 (via Ollama) for:

Summarization

Question answering with references

Display results in Streamlit UI.

⚙️ Tech Stack

Python

LangChain

ChromaDB

Hugging Face Embeddings

Ollama (LLaMA3:8B instruct quantized model)

Streamlit

PyMuPDF / pdfplumber

pytesseract + pdf2image

▶️ How to Run

Clone this repository:

git clone <your-repo-link>
cd pdf-summarizer-qa


Install dependencies:

pip install -r requirements.txt


Make sure Ollama is installed and model is pulled:

ollama pull llama3:8b-instruct-q4_K_M


Run the app:

streamlit run app.py


Upload a PDF → Ask questions → Get answers with page citations.
