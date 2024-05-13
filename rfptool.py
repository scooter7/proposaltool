import streamlit as st
import os
import io
import requests
from PyPDF2 import PdfFileReader
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.chat_models import ChatOpenAI  # Corrected import based on deprecation warning

# Initialize the LangChain OpenAI Chat model with the API key from Streamlit secrets
chat_model = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-0125",
    openai_api_key=st.secrets["openai"]["api_key"]
)

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []

    def add_document(self, text):
        self.documents.append(text)
        embedding = self.model.encode(text, convert_to_tensor=True)
        self.embeddings.append(embedding)

    def query(self, text):
        query_embedding = self.model.encode(text, convert_to_tensor=True)
        distances = np.inner(query_embedding, np.vstack(self.embeddings))
        most_similar_idx = np.argmax(distances)
        return self.documents[most_similar_idx]

def read_pdf(file):
    """Read pages from a PDF file and return them as a list of strings."""
    reader = PdfFileReader(file)
    text = [reader.getPage(i).extract_text() for i in range(reader.numPages)]
    return text

def download_github_files(base_url, local_dir='./rfps/'):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    response = requests.get(base_url)
    if response.status_code == 200:
        files = response.json()
        for file in files:
            if 'download_url' in file and file['name'].endswith('.pdf'):
                pdf_response = requests.get(file['download_url'])
                with open(os.path.join(local_dir, file['name']), 'wb') as f:
                    f.write(pdf_response.content)

def main():
    st.title("RAG Proposal Tool with LangChain")

    vector_store = VectorStore()
    download_github_files("https://api.github.com/repos/scooter7/proposaltool/contents/rfps")
    local_dir = './rfps/'
    
    for filename in os.listdir(local_dir):
        if filename.endswith('.pdf'):
            with open(os.path.join(local_dir, filename), 'rb') as f:
                pdf_text = read_pdf(f)
                vector_store.add_document(" ".join(pdf_text))

    uploaded_file = st.file_uploader("Upload your RFP document", type=['pdf'])
    
    if uploaded_file is not None:
        document_text = read_pdf(uploaded_file)
        st.write("Uploaded RFP Text:")
        st.write(document_text)
        
        requirements = " ".join(document_text)
        st.write("Requirements Outline:")
        st.write(requirements)

        query_results = vector_store.query(requirements)
        st.write("Related content from past proposals:")
        st.write(query_results)

        response = chat_model.complete(
            prompt=f"Generate a proposal based on: {requirements} and similar past proposal: {query_results}",
            max_tokens=1024
        )
        st.write("Generated Proposal:")
        st.write(response['choices'][0]['text'])

if __name__ == "__main__":
    main()
