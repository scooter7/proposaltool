import os
from typing import List, Optional
from operator import itemgetter

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough
from langchain.schema import format_document, Document
from langchain.schema.runnable import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
    RunnableMap,
)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader

st.title("Vector Store Proposal Tool")

# Initialize OpenAI models
model = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-0125",
    openai_api_key=st.secrets["openai"]["api_key"],
)
embedding = OpenAIEmbeddings(openai_api_key=st.secrets["openai"]["api_key"])

# Define paths for vector stores
politic_vector_store_path = "politic_vector_store_path.faiss"
environmental_vector_store_path = "environmental_vector_store_path.faiss"

# Function to load vector stores
def load_vector_store(file_path, embedding):
    if os.path.exists(file_path):
        return FAISS.load_local(file_path, embedding, allow_dangerous_deserialization=True)
    return None

# Load vector stores if they exist
politic_vector_store = load_vector_store(politic_vector_store_path, embedding)
environmental_vector_store = load_vector_store(environmental_vector_store_path, embedding)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    """Combine documents into a single string."""
    return document_separator.join([format_document(doc, document_prompt) for doc in docs])

# Function to format chat history
def format_chat_history(chat_history: dict) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        actor = "Human" if dialogue_turn["role"] == "user" else "Assistant"
        buffer += f"{actor}: {dialogue_turn['content']}\n"
    return buffer

# Sidebar configuration for vector store roles
st.sidebar.header("Vector Store Settings")
informing_store_name = st.sidebar.selectbox(
    "Select the Informing Vector Store:",
    ["Politic", "Environmental"],
    index=0
)
target_store_name = "Environmental" if informing_store_name == "Politic" else "Politic"

# Setup to retrieve information from vector stores based on user queries
class ConfigurableFaissRetriever(RunnableSerializable[str, List[Document]]):
    vector_store_topic: str

    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> List[Document]:
        vector_store = politic_vector_store if self.vector_store_topic == "Politic" else environmental_vector_store
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return retriever.invoke(input, config=config)

# Main interaction: Generate a proposal
st.header("Generate Proposal")
st.write(f"Informing Store: {informing_store_name}, Target Store: {target_store_name}")

# User input for requirements or queries
user_query = st.text_area("Enter your requirements or query here:")

if st.button("Generate Proposal"):
    if user_query:
        # Retrieve context from the Informing vector store (Politic)
        informing_retriever = ConfigurableFaissRetriever(vector_store_topic=informing_store_name)
        retrieved_docs_informing = informing_retriever.invoke(user_query)
        context_from_informing_store = combine_documents(retrieved_docs_informing)

        # Retrieve requirements from the Target vector store (Environmental)
        target_retriever = ConfigurableFaissRetriever(vector_store_topic=target_store_name)
        retrieved_docs_target = target_retriever.invoke(user_query)
        requirements_from_target_store = combine_documents(retrieved_docs_target)

        # Display context information
        st.subheader("Context from Informing Store (Politic)")
        st.write(context_from_informing_store)

        st.subheader("Requirements from Target Store (Environmental)")
        st.write(requirements_from_target_store)

        # Prepare a prompt to generate a proposal
        response_prompt = (
            f"Context from {informing_store_name} Store:\n{context_from_informing_store}\n\n"
            f"Use this context to craft a proposal addressing these requirements from the {target_store_name} Store:\n"
            f"{requirements_from_target_store}"
        )

        # Generate the proposal using the model
        response = model.invoke(input={"prompt": response_prompt, "max_tokens": 500})

        # Display the proposal
        st.subheader("Crafted Proposal")
        st.write(response['choices'][0]['message']['content'])
    else:
        st.warning("Please enter some requirements to generate a proposal.")

# File uploaders for updating vector stores
with st.expander("Update Vector Stores"):
    politic_index_uploaded_file = st.file_uploader(
        "Upload a PDF file to update the Politic vector store:", type="pdf", key="politic_index"
    )
    if politic_index_uploaded_file is not None:
        pdf_reader = PdfReader(politic_index_uploaded_file)
        text_data = "\n\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        split_data = text_data.split("\n\n")
        politic_vector_store = FAISS.from_texts(split_data, embedding=embedding)
        politic_vector_store.save_local(politic_vector_store_path)
        st.success("Politic vector store updated successfully!")

    environmental_index_uploaded_file = st.file_uploader(
        "Upload a PDF file to update the Environmental vector store:", type="pdf", key="environmental_index"
    )
    if environmental_index_uploaded_file is not None:
        pdf_reader = PdfReader(environmental_index_uploaded_file)
        text_data = "\n\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        split_data = text_data.split("\n\n")
        environmental_vector_store = FAISS.from_texts(split_data, embedding=embedding)
        environmental_vector_store.save_local(environmental_vector_store_path)
        st.success("Environmental vector store updated successfully!")

# Chat history and message interaction
st.header("Chat with Vector Stores")
if os.path.exists(politic_vector_store_path) or os.path.exists(environmental_vector_store_path):
    vector_store_topic = st.selectbox(
        "Choose Vector Store for Interaction:",
        options=["Politic", "Environmental"],
        index=0
    )
    output_type = st.selectbox(
        "Select the Output Type:",
        options=["detailed", "single_line"],
        index=0
    )

    if "message" not in st.session_state:
        st.session_state["message"] = [{"role": "assistant", "content": "Hello 👋, How can I assist you?"}]

    for message in st.session_state.message:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask me anything"):
        st.session_state.message.append({"role": "user", "content": query})

        # Context and response generation based on interaction
        interaction_retriever = ConfigurableFaissRetriever(vector_store_topic=vector_store_topic)
        interaction_docs = interaction_retriever.invoke(query)
        interaction_context = combine_documents(interaction_docs)

        chat_response_prompt = f"Context: {interaction_context}\nAnswer this question:\n{query}"
        chat_response = model.invoke(input={"prompt": chat_response_prompt, "max_tokens": 150})

        st.session_state.message.append({"role": "assistant", "content": chat_response['choices'][0]['message']['content']})
        with st.chat_message("assistant"):
            st.markdown(chat_response['choices'][0]['message']['content'])
