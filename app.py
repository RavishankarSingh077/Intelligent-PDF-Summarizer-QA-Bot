import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# PDF + OCR
import fitz  # PyMuPDF (for extracting text)
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# LangChain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Streamlit UI
st.title("Intelligent PDF Summarizer & QA Bot")

uploaded_file = st.file_uploader("Upload a PDF (report / research paper)", type="pdf")
query = st.text_input("Ask a question about the PDF")


# Step 1: Extract text from PDF (with OCR fallback)

def extract_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []
    ocr_pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        if len(text.strip()) < 30:  # अगर text बहुत कम है तो OCR से निकालेंगे
            images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
            if images:
                ocr_text = pytesseract.image_to_string(images[0])
                if len(ocr_text.strip()) > 0:
                    text = ocr_text
                    ocr_pages.append(i+1)
        pages_text.append(text)

    return pages_text, ocr_pages



# Step 2: Process PDF only once

if uploaded_file and "retriever" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Extract text with OCR fallback
    st.info("Extracting text (OCR fallback if needed)...")
    pages_text, ocr_pages = extract_text_with_ocr(pdf_path)

    # Save extracted text (Deliverable 1)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/extracted_text.txt", "w", encoding="utf-8") as f:
        for i, text in enumerate(pages_text, start=1):
            f.write(f"\n===== Page {i} =====\n{text}\n")
    st.success("✔ Extracted text saved to outputs/extracted_text.txt")
    if ocr_pages:
        st.warning(f"OCR used on pages: {ocr_pages}")

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = [{"page_content": text, "metadata": {"page": i+1}} for i, text in enumerate(pages_text)]
    from langchain.schema import Document
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in documents]
    texts = text_splitter.split_documents(docs)

    # Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embedding=embeddings)
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    st.session_state.texts = texts
    st.success("✔ Vector store ready")


# Step 3: Initialize LLaMA3 (Ollama)

if "llm" not in st.session_state:
    st.session_state.llm = OllamaLLM(model="llama3:8b-instruct-q4_K_M")
    st.success("✔ LLaMA3 model loaded")


# Step 4: Generate Structured Summary

if uploaded_file and "summary" not in st.session_state:
    st.subheader("Structured Summary")
    if st.button("Generate Summary"):
        context = "\n".join([doc.page_content for doc in st.session_state.texts[:5]])
        prompt = f"""
        Summarize the following PDF text into a structured, hierarchical summary 
        with sections, bullet points, and key insights:

        {context}
        """
        summary = st.session_state.llm.invoke(prompt)
        st.session_state.summary = summary
        st.write(summary)

        with open("outputs/summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        st.success("✔ Summary saved to outputs/summary.txt")


# Step 5: Q/A Interface

if query and "retriever" in st.session_state and "llm" in st.session_state:
    qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        retriever=st.session_state.retriever,
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})

    st.subheader("Answer:")
    st.write(result["result"])

    st.subheader("Sources (with page numbers):")
    for doc in result["source_documents"]:
        st.markdown(f"- Page {doc.metadata.get('page', '?')} → {doc.page_content[:200]}...")


# Step 6: Sample Queries

if uploaded_file and st.button("Run Sample Queries"):
    st.subheader("Sample Queries with Page-Cited Answers")
    sample_qs = [
        "Give me a high-level summary of this document.",
        "What are the key findings or contributions?",
        "Which section discusses methodology? Cite the page.",
        "What are the limitations mentioned?",
    ]
    qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        retriever=st.session_state.retriever,
        return_source_documents=True
    )
    for q in sample_qs:
        st.markdown(f"**Q:** {q}")
        result = qa_chain.invoke({"query": q})
        st.write(result["result"])
        st.write("**Sources:** " + ", ".join([f"p.{doc.metadata.get('page','?')}" for doc in result["source_documents"]]))
        st.markdown("---")


