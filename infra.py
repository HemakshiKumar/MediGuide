import os
# Prevent transformers from loading tensorflow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import requests
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Any, List, Optional

# Import configurations and token from config.py
from config import HF_API_TOKEN, MODEL_NAME

st.set_page_config(
    page_title="MediGuide",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------------------------------------
# 1. Hugging Face Router Query Function
# -------------------------------------------------------------
def query_llm(prompt: str) -> str:
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.2
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API Error ({response.status_code}): {response.text}")
        
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

# -------------------------------------------------------------
# 2. Embeddings & Vector Database
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource(show_spinner="Loading Medical Knowledge Base...")
def get_database():
    embedding_model = get_embeddings_model()
    faiss_index_path = os.path.join(os.path.dirname(__file__), "storedata", "db_faiss")
    if not os.path.exists(faiss_index_path):
        return None
    database = FAISS.load_local(
        faiss_index_path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    return database

def get_medical_answer(question: str) -> str:
    cleaned = question.strip().lower().rstrip("!.,?")
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings", "hi there", "hello there"}
    
    if cleaned in greetings:
        return "Hello! I am MediGuide, your health assistant. How can I help you today? Feel free to ask any medical or health-related question."

    db = get_database()
    if db is None:
        return "⚠️ Medical knowledge base is not loaded."
    
    # Retrieve top relevant context chunks from FAISS
    docs = db.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are MediGuide, a helpful and intelligent health chatbot.

Instructions:
- If the user is asking a general greeting, polite check-in, or question about your capabilities, respond politely and invite them to ask their health question.
- For medical questions: Use the provided medical context to give a clear, comprehensive, and well-explained answer in 2 well-structured paragraphs (covering definitions, causes, symptoms, or treatments mentioned in the context).
- If the question is medical but cannot be answered from the context, state: "I don't have enough information in the medical guide to answer that."

Context:
{context}

Question:
{question}

Answer:"""

    return query_llm(prompt)



# -------------------------------------------------------------
# 3. Main Streamlit User Interface
# -------------------------------------------------------------
def main():
    st.title("MediGuide")
    st.caption("Ask any health-related question and get answers.")

    # Medical Disclaimer Banner
    st.warning(
        "**Disclaimer:** This chatbot is for general informational purposes only and may not always be accurate or complete. "
        "Always consult a qualified healthcare professional for medical concerns, and seek immediate medical attention in an emergency."
    )

    # Verify vector store existence
    faiss_index_path = os.path.join(os.path.dirname(__file__), "storedata", "db_faiss")
    if not os.path.exists(faiss_index_path):
        st.error("⚠️ Vector database not found in `storedata/db_faiss`. Please run `py -3.12 .\\build_index.py` first.")
        return

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    user_prompt = st.chat_input("Your health question (e.g., What are the symptoms of migraine?):")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    answer = get_medical_answer(user_prompt)
                    
                    if "answer:" in answer.lower():
                        answer = answer[answer.lower().find("answer:") + 7:].strip()

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
