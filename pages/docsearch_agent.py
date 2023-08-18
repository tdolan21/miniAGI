from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import PGEmbedding
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from pathlib import Path
import os
from langchain.vectorstores.pgvector import PGVector
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI


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


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    docs_with_score = store.similarity_search_with_score(prompt)
    with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            docs_with_score = db.similarity_search_with_score(prompt)
            st.write(prompt)
            st.write(docs_with_score)