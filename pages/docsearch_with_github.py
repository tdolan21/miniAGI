from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
import os
import streamlit as st
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from pathlib import Path
from langchain.callbacks import StreamlitCallbackHandler



loader = DirectoryLoader(Path(os.getenv("DOCUMENTS_PATH")))
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()



CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "miniAGI"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "Royals21"),
)
COLLECTION_NAME = "documentation"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

st.title("miniAGI :computer:")
st.subheader("PGVector Document Search")
st.write("Load files to PostgreSQL by putting them into the `documents` folder. Then, click the button below to load them into the database.")


# Create columns for repo path, repo name, and process button
col1, col2, col3 = st.columns(3)
# Clone
repo_input = col1.text_input("Enter repository path:", value="documents/repositories")
# Input for repository name
repo_name = col2.text_input("Enter repository link:", value="")

process_button = col3.button("Process")







# If the process button is clicked, execute the logic
if process_button:
    # Clone
    repo_path = os.path.join(repo_input, repo_name.split("/")[-1])
    repo = Repo.clone_from(repo_name, to_path=repo_path)

    # Load
    loader = GenericLoader.from_filesystem(
        repo_path+"/",
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    len(documents)

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                chunk_size=2000, 
                                                                chunk_overlap=200)
    texts = python_splitter.split_documents(documents)
    len(texts)
    


from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8},
)
llm = ChatOpenAI(model_name="gpt-4") 
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)




if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    result = qa(prompt)
    with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            st.write(prompt)

            st.write(result['answer'])
            
            




