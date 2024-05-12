import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_openai import OpenAI, OpenAIEmbeddings  # Ensure these are from langchain_openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.chains import ConversationalRetrievalChain  # Assuming chains are part of langchain_openai now
from langchain_openai.callbacks import get_openai_callback  # Updated for langchain_openai
from langchain.schema import Document  # Adjust if this also has a new path

##########################################################################
## DEFINE VARIABLES
# Directories and file extensions
SUB_EXT = 'rfps'  # Directory where PDFs are stored
SUB_EMB = 'SUB_EMB'
EXT = '.pdf'
EMB_EXT = '.pkl'

# Initialize OpenAI LLM and embeddings using Streamlit secrets:
openai_api_key = st.secrets["OPENAI_API_KEY"]
if openai_api_key:
    llm = OpenAI(api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
else:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY in your Streamlit secrets.")

##########################################################################
## DEFINE FUNCTIONS

def f_scan_directory_for_ext(directory, extension):
    """Scan the specified directory for files ending with the given extension."""
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def f_create_embedding(new_file_trunk, new_file_pdf_path, file_persistent_dir_path):
    """Create text embeddings for the given PDF and save them using FAISS."""
    print(f"Creating embedding for {file_persistent_dir_path}")
    pdf_reader = PdfReader(new_file_pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle None return from extract_text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chunks = text_splitter.split_text(text=text)
    
    # Convert chunks into embeddings
    embeddings_list = [embeddings.embed_text(chunk) for chunk in chunks]  # Updated method name
    embeddings_matrix = np.vstack(embeddings_list)
    
    # Use FAISS for vector storage
    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_matrix)
    
    # Persist the FAISS index to disk
    faiss.write_index(index, os.path.join(file_persistent_dir_path, 'faiss_index'))

def main():
    """Main function to initialize the Streamlit app and process user interactions."""
    st.title('PDF Chatbot App')

    # Ensure the SUB_EXT directory exists to prevent os.listdir errors
    if not os.path.exists(SUB_EXT):
        os.makedirs(SUB_EXT)
        st.warning(f"The directory {SUB_EXT} was created as it did not exist.")
    
    # Dynamically list all PDFs in the specified directory
    files_in_directory = f_scan_directory_for_ext(SUB_EXT, EXT)

    # Create a Streamlit sidebar with checkboxes to select PDFs
    with st.sidebar:
        st.title('Step1: Select PDFs')
        st.markdown('''
        This app is an LLM-powered chatbot allowing
        you to load one to all PDFs located in a 
        dedicated sub-directory into a vector store
        so that you can have Q&A-sessions with the
        combined selected content.
        ''')
        add_vertical_space(20)

        selected_files = []
        for file in files_in_directory:
            if st.sidebar.checkbox(file, key=file):
                selected_files.append(file)

    # Display selected files or perform actions based on selection
    st.write('Selected Files (Result of Step1):')
    for file in selected_files:
        st.write(file)

    l_db_pathes_to_load = ["No confirmed selection yet!"]
    if st.button('Step2: Proceed to chat with selected files'):
        st.session_state['selected_files'] = selected_files
        l_db_pathes_to_load = [os.path.join(SUB_EMB, filename[:-len(EXT)]) for filename in st.session_state['selected_files']]
        for pathname in l_db_pathes_to_load:
            st.write(f"Selected: {pathname}")

    st.header("Chat with the PDFs of your choice")
    DB_final = None
    if l_db_pathes_to_load == ["No confirmed selection yet!"]:
        for pathname in l_db_pathes_to_load:
            st.write(pathname)
    elif len(l_db_pathes_to_load) == 0:
        st.write("At least 1 file must be selected")
    else:
        all_embeddings = []
        all_ids = []
        for db_path in l_db_pathes_to_load:
            # Load the FAISS index from disk
            index_path = os.path.join(db_path, 'faiss_index')
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
                # Mock Document: For demonstration, associate embeddings with their vector ids
                for i in range(index.ntotal):
                    all_embeddings.append(index.reconstruct(i))
                    all_ids.append(f"{db_path}_{i}")
        
        # Combine all embeddings into one large FAISS index for searching
        if all_embeddings:
            dim = all_embeddings[0].shape[0]
            DB_final = faiss.IndexFlatL2(dim)
            DB_final.add(np.array(all_embeddings))

    with st.form("query_input"):
        query = st.text_input("Step3: Ask questions about the selected PDF file (or enter EXIT to exit):")
        submit_button = st.form_submit_button("Submit Query")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if query != "EXIT":
        if submit_button and DB_final:
            st.write(f"Your query was: {query}")
            query_vector = embeddings.embed_text([query])[0]  # Ensure this is the right method for embeddings
            D, I = DB_final.search(np.array([query_vector]), k=4)
            
            # Fetch and display answers - For demonstration, just show distances and ids
            st.write("Closest segments to your query based on the selected PDFs:")
            for i, idx in enumerate(I[0]):
                st.write(f"Doc {all_ids[idx]} with distance {D[0][i]}")
            
            chat_tuple = (query, [all_ids[idx] for idx in I[0]])
            st.session_state['chat_history'].append(chat_tuple)
            add_vertical_space(20)
    else:
        st.warning('You chose to exit the chat.')
        st.stop()

    if 'chat_history' in st.session_state:
        p_chat_history = [entry for entry in st.session_state['chat_history']]
        for entry in p_chat_history:
            print('--------------')
            print(entry)

# Process new files and create embeddings
if __name__ == '__main__':
    main()
