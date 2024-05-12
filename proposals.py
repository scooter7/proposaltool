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
# Change these to match your specific directory and file extension
SUB_EXT = 'rfps'
SUB_EMB = 'SUB_EMB'
EXT = '.pdf'
EMB_EXT = '.pkl'
FILE_LIST = 'file_name_list.txt'

#_________________________________________
# Initialize openAI llm and embeddings using Streamlit secrets:
openai_api_key = st.secrets["OPENAI_API_KEY"]
if openai_api_key:
    llm = OpenAI(openai_api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Use llm in your Streamlit app
else:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY in your Streamlit secrets.")
##########################################################################
## DEFINE FUNCTIONS

def f_scan_directory_for_ext(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def f_get_existing_files(file_name_list):
    with open(file_name_list, 'r') as file:
        return set(file.read().splitlines())

def f_update_file_list(file_name_list, new_files):
    with open(file_name_list, 'a') as file:
        for new_file in new_files:
            file.write(new_file + '\n')

def f_create_embedding(new_file_trunk, new_file_pdf_path, file_persistent_dir_path):
    # Dummy function - replace with actual embedding logic
    print(f"Creating embedding for {file_persistent_dir_path}")
    pdf_reader = PdfReader(new_file_pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chunks = text_splitter.split_text(text=text)
    # workaround to stupid bug ?!? -> https://github.com/langchain-ai/langchain/issues/2877
    new_chunks = [Document(page_content=chunk) for chunk in chunks]
    db = Chroma.from_documents(
        documents=new_chunks, embedding=embeddings, persist_directory=file_persistent_dir_path)
    db.persist()

def main():
    # Initialize the Streamlit app
    st.title('PDF Chatbot App')
    # Step 4: Create a Streamlit sidebar with checkboxes
    with st.sidebar:
        st.title('Step1: Select PDFs')
        st.markdown('''
        This app is an LLM-powered chatbot allowing
        you to load one to all PDFs located in a 
        dedicated sub-directory into a vector store
        so that you can have Q&A-sessions with the
        combined selected content.
        ''')
        add_vertical_space(0)

        selected_files = []
        for file in files_in_directory:
            if st.sidebar.checkbox(file, key=file):
                selected_files.append(file)

    # Display selected files or perform actions based on selection
    st.write('Selected Files (Result of Step1):')
    for file in selected_files:
        st.write(file)

    l_db_pathes_to_load = ["No confirmed selection yet!"]
    # Button to signal the end of the selection process
    if st.button('Step2: Proceed to chat with selected files'):
        st.session_state['selected_files'] = selected_files
        l_db_pathes_to_load = [os.path.join(SUB_EMB, filename[:-len(EXT)]) for filename in st.session_state['selected_files']]
        for pathname in l_db_pathes_to_load:
            st.write(f"Selected: {pathname}")

    st.header("Chat with the PDFs of your choice")
    # Create a new, empty Chroma object to receive input based on the previous document selection
    DB_final = Chroma(embedding_function=embeddings)
    # loop to load all Chroma embedding databases of selected files from disk to vector store
    if l_db_pathes_to_load == ["No confirmed selection yet!"]:
        for pathname in l_db_pathes_to_load:
            st.write(pathname)
    elif len(l_db_pathes_to_load) == 0:
        st.write("At least 1 file must be selected")
    else:
        for db_path in l_db_pathes_to_load:
            # Load the embeddings from the existing vector databases
            DB_aux = Chroma(persist_directory=db_path, embedding_function=embeddings)
            DB_aux_data = DB_aux._collection.get(include=['documents','metadatas','embeddings'])
            DB_final._collection.add(
                 embeddings=DB_aux_data['embeddings'],
                 metadatas=DB_aux_data['metadatas'],
                 documents=DB_aux_data['documents'],
                 ids=DB_aux_data['ids'])

    # Accept user questions/query via a button-confirmed form:
    with st.form("query_input"):
        query = st.text_input("Step3: Ask questions about the selected PDF file (or enter EXIT to exit):")
        submit_button = st.form_submit_button("Submit Query")

    # Initialize session_state if it doesn't exist
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
            add_vertical_space(1)
            st.write(f"The response: {response['answer']}")
            add_vertical_space(1)
            # Update chat_history in session_state
            chat_tuple = (query, response['answer'])
            st.session_state['chat_history'].append(chat_tuple)
    else:
        st.warning('You chose to exit the chat.')
        st.stop()
    # Print chat history to terminal (no GUI)
    if 'chat_history' not in st.session_state:
        pass
    else:
        p_chat_history = [entry for entry in st.session_state['chat_history']]
        for entry in p_chat_history:
            print('--------------')
            print(entry)
##########################################################################
## MAIN PROGRAM
# Step 1: Scan the SUB_EXT directory
files_in_directory = f_scan_directory_for_ext(SUB_EXT, EXT)

# Step 2: Check against the list in file_name_list.txt
known_files = f_get_existing_files(FILE_LIST)
new_files = [f for f in files_in_directory if f not in known_files]
new_files_trunk = [f[:-len(EXT)] for f in new_files ]

# Step 3: Process new files
# Path to the SUB_EMB directory
SUB_EMB_dir = os.path.join(SUB_EMB)
# Iterate over the list of names
for name in new_files_trunk:
    # Construct the path to the sub-directory
    subdir_path = os.path.join(SUB_EMB_dir, name)
    # Check if the sub-directory exists
    if not os.path.exists(subdir_path):
        # If not, create it
        os.makedirs(subdir_path)
for new_file in new_files:
    new_file_trunk = new_file[:-len(EXT)]
    # f_create_embedding(new_file_trunk, os.path.join(SUB_EXT, new_file), os.path.join(os.getcwd(), SUB_EMB, new_file_trunk))
    f_create_embedding(new_file_trunk, os.path.join(SUB_EXT, new_file), os.path.join(SUB_EMB, new_file_trunk))
f_update_file_list(FILE_LIST, new_files)

# Step 4: call main function that contains all the rest
if __name__ == '__main__':
    main()
