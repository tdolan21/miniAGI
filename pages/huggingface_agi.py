from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import PromptTemplate, LLMChain
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import psycopg2
import os


load_dotenv()

def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("PGVECTOR_DATABASE"),
        user=os.getenv("PGVECTOR_USER"),
        password=os.getenv("PGVECTOR_PASSWORD"), 
        host=os.getenv("PGVECTOR_HOST")
    )

# Fetch saved templates from database
def fetch_templates():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, template FROM prompt_templates;")
    templates = cur.fetchall()
    cur.close()
    conn.close()
    return templates
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore
def get_conversation_chain(vectorstore):
    
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
    

# Main Page
templates = fetch_templates()
# Place the selectbox in the sidebar
selected_template = st.sidebar.selectbox("Select a template", templates, format_func=lambda x: x[1])
with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


question = ""



prompt = PromptTemplate(template=selected_template[2], input_variables=["question"])

# Title and info popover
st.title("Huggingface models :rocket:")
st.info("This is a search engine for several hugging face models. This section is completely free, but requires a large amount of GPU compute.  Please refer to the documentation if you have any questions.")


# User select a model from huggingface 
# Text generation models only, no text2text models
# See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
repo_id = st.selectbox("Select a model",
                       ("tiiuae/falcon-7b", "Qwen/Qwen-7B" ))  
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 250}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
          

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    question = prompt
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
       
        st.write(llm_chain.run(question))   
    
        
           