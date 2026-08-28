# 🩺 MediGuide – Intelligent Health Chatbot

MediGuide is an intelligent medical assistant built with **Streamlit**, **LangChain**, and **FAISS**, using Retrieval-Augmented Generation (RAG) to provide fast, reliable, context-grounded answers to medical queries.

---

## 🚀 Features

- 🧠 **Retrieval-Augmented Generation (RAG)**: Extracts precise medical context from loaded PDF documents.
- ⚡ **Zero-Lag Cloud Inference**: Uses Hugging Face's Free Serverless Inference API for sub-second responses.
- 🔍 **FAISS Vector Store**: Fast semantic search using `sentence-transformers/all-MiniLM-L6-v2`.
- 🖥️ **Streamlit UI**: Clean and interactive chat interface with token configuration in the sidebar.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Vector Store & Indexing | `FAISS` |
| Text Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Text Splitter | `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 50) |
| RAG Framework | `LangChain` (`RetrievalQA`) |
| LLM Backend | Hugging Face Serverless API (`mistralai/Mistral-7B-Instruct-v0.3`) |
| Frontend | `Streamlit` |

---

## 📦 Installation & Setup

1. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add medical PDFs**:
   Create a `book/` folder in the project directory and place your medical PDF files inside it:
   ```bash
   mkdir book
   # Copy your medical PDF file(s) into book/
   ```

3. **Build the FAISS Vector Database**:
   ```bash
   python build_index.py
   ```
   *This reads all PDFs from `book/`, computes chunk embeddings, and saves the index to `storedata/db_faiss`.*

4. **Launch the Streamlit App**:
   ```bash
   streamlit run infra.py
   ```

5. **Enter your Hugging Face API Token**:
   - Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Paste it into the sidebar input field in the web app, or set the environment variable:
     ```bash
     export HUGGINGFACEHUB_API_TOKEN="your_hf_token_here"
     ```

---

## 📁 Directory Structure

```bash
MediGuide/
├── book/                   # Medical PDFs (User provided)
├── storedata/              # Generated FAISS vector index
├── build_index.py          # Script to parse PDFs and build FAISS index
├── infra.py                # Fast Streamlit chat application
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
└── LICENSE
```

---

## ⚠️ Disclaimer
MediGuide is designed for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
