# 🩺 MediGuide – Intelligent Health Chatbot
-
MediGuide is an intelligent medical assistant built with Streamlit and a powerful QA chain backend, designed to provide users with reliable and easy-to-understand answers to health-related questions. It simulates a conversational interface where users can input queries in natural language and receive medically-informed responses in real-time.
---

## Features

- 🧠 **Symptom Understanding**: Detects health-related intents and interprets user symptoms.
- 📚 **Context-Aware QA**: Uses retrieval-augmented generation (RAG) to find the most relevant information from medical PDFs.
- ⚙️ **Hugging Face Transformers**: Leverages pre-trained models for compact yet powerful natural language understanding.
- 🔍 **FAISS Indexing**: Efficient document storage and semantic search over medical texts.
- 🖥️ **Streamlit UI**: Simple, interactive chat interface for asking questions in natural language.

---

## Tech Stack

| Component              | Stack / Tool                        |
|------------------------|-------------------------------------|
| Language Model         | `TinyLlama/TinyLlama-1.1B-Chat`     |
| Embedding Model        | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store           | `FAISS`                             |
| QA Pipeline            | `LangChain`                         |
| UI Framework           | `Streamlit`                         |
| PDF Parsing            | `LangChain PyPDFLoader`             |

---

## How It Works

1. **PDF Loader**: Medical documents are loaded and split into chunks.
2. **Embedding + Indexing**: Chunks are embedded using `MiniLM` and indexed in FAISS.
3. **User Query**: Chat input is processed in Streamlit.
4. **RAG Pipeline**: Top relevant chunks are retrieved, passed to the LLM for generation.
5. **Answer**: A concise, helpful, context-grounded answer is returned to the user.

---

## Installation

```bash
!pip install streamlit
!pip install langchain
!pip install langchain_community
!pip install langchain_huggingface
!pip install transformers
!pip install torch
!pip install pypdf
!pip install faiss-cpu
```

## Running the app
```bash
streamlit run infra.py
```

### Directory Structure
```bash
mediguide/
├── book/                   # Medical PDFs
├── storedata/              # FAISS index storage
├── infra.py                # Streamlit app code
├── requirements.txt
└── README.md
```
## Disclaimer
MediGuide is not a replacement for professional medical advice. It is intended only for informational and educational purposes in non-emergency scenarios.

## License
This project is open-source and available under the MIT License.

-
Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
---
