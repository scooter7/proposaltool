from PyPDF2 import PdfReader
import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
from langchain.schema import Document

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
    llm = OpenAI(openai_api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
else:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY in your Streamlit secrets.")
##########################################################################
## DEFINE FUNCTIONS

def f_scan_directory_for_ext(directory, extension):
    """Scan the specified directory for files ending with the given extension."""
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def f_create_embedding(new_file_trunk, new_file_pdf_path, file_persistent_dir_path):
    """Create text embeddings for the given PDF and save them in the specified directory."""
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
    new_chunks = [Document(page_content=chunk) for chunk in chunks]
    db = Chroma.from_documents(
        documents=new_chunks, embedding=embeddings, persist_directory=file_persistent_dir_path)
    db.persist()

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
        add_vertical_space(20)  # Adjust vertical space as needed

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
    DB_final = Chroma(embedding_function=embeddings)
    if l_db_pathes_to_load == ["No confirmed selection yet!"]:
        for pathname in l_db_pathes_to_load:
            st.write(pathname)
    elif len(l_db_pathes_to_load) == 0:
        st.write("At least 1 file must be selected")
    else:
        for db_path in l_db_pathes_to_load:
            DB_aux = Chroma(persist_directory=db_path, embedding_function=embeddings)
            DB_aux_data = DB_aux._collection.get(include=['documents','metadatas','embeddings'])
            DB_final._collection.add(
                 embeddings=DB_aux_data['embeddings'],
                 metadatas=DB_aux_data['metadatas'],
                 documents=DB_aux_data['documents'],
                 ids=DB_aux_data['ids'])

    with st.form("query_input"):
        query = st.text_input("Step3: Ask questions about the selected PDF file (or enter EXIT to exit):")
        submit_button = st.form_submit_button("Submit Query")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if query != "EXIT":
        if submit_button:
            st.write(f"Your query was: {query}")
            retriever = DB_final.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)
            with get_openai_callback() as cb:
                response = chain({'question': query, 'chat_history': st.session_state['chat_history']})
                print(cb)
            add_vertical_space(20)
            st.write(f"The response: {response['answer']}")
            chat_tuple = (query, response['answer'])
            st.session_state['chat_history'].append(chat_tuple)
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
