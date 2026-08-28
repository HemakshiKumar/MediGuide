import os
# Prevent transformers from loading tensorflow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import requests
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional
from pydantic import Field

# Import configurations and token from config.py
from config import HF_API_TOKEN, MODEL_NAME

st.set_page_config(
    page_title="MediGuide",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------------------------------------
# 1. Custom Hugging Face Router LLM
# -------------------------------------------------------------
class HFRouterLLM(LLM):
    token: str = Field(default_factory=lambda: HF_API_TOKEN)
    model_name: str = MODEL_NAME

    @property
    def _llm_type(self) -> str:
        return "huggingface_router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
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

@st.cache_resource(show_spinner="Connecting to Model...")
def get_llm():
    return HFRouterLLM(token=HF_API_TOKEN, model_name=MODEL_NAME)

def set_custom_prompt(custom_prompt_template: str):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

@st.cache_resource(show_spinner=False)
def get_qa_chain():
    db = get_database()
    if db is None:
        return None
    
    prompt_template = """Use the following medical context to provide a clear, comprehensive, and well-explained answer to the question.
Provide a detailed response in 2 well-structured paragraphs (covering definitions, causes/symptoms, or relevant medical details found in the context).
If the information is not present in the context, clearly state "I don't have enough information in the medical guide to answer that."

Context:
{context}

Question:
{question}

Answer:"""

    prompt = set_custom_prompt(prompt_template)
    llm_instance = get_llm()
    
    return RetrievalQA.from_chain_type(
        llm=llm_instance,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt}
    )

def format_answer(text: str) -> str:
    return text.strip()

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
                    qa_chain = get_qa_chain()
                    if qa_chain is None:
                        st.error("Failed to initialize QA Chain. Ensure database is built.")
                        return

                    response = qa_chain.invoke({"query": user_prompt})
                    raw_result = response.get("result", "")
                    
                    if "answer:" in raw_result.lower():
                        ans_to_edit = raw_result[raw_result.lower().find("answer:") + 7:].strip()
                    else:
                        ans_to_edit = raw_result.strip()

                    ans_formatted = format_answer(ans_to_edit)
                    st.markdown(ans_formatted)
                    st.session_state.messages.append({"role": "assistant", "content": ans_formatted})

                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
