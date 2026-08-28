import os
# Prevent transformers from trying to load tensorflow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_db():
    pdf_dir = "book/"
    if not os.path.exists(pdf_dir):
        print(f"Directory '{pdf_dir}' does not exist. Please create it and add your medical PDFs.")
        return

    print("Loading medical PDF documents from book/ ...")
    loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    data = loader.load()
    print(f"Loaded {len(data)} pages from PDFs.")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    data_chunks = text_splitter.split_documents(data)
    print(f"Created {len(data_chunks)} chunks.")

    print("Generating embeddings with sentence-transformers/all-MiniLM-L6-v2 ...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Building and saving FAISS index (this may take a few minutes for ~9.7k chunks)...")
    os.makedirs("storedata/db_faiss", exist_ok=True)
    database = FAISS.from_documents(data_chunks, embedding_model)
    database.save_local("storedata/db_faiss")
    print("FAISS vector database saved successfully to 'storedata/db_faiss'!")

if __name__ == "__main__":
    build_vector_db()
