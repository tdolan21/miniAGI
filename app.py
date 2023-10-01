from langchain import VectorDBQA
from langchain.agents import AgentType, initialize_agent, load_tools, Tool, Agent
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
import os
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.memory import PostgresChatMessageHistory
from langchain.utilities.arxiv import ArxivAPIWrapper
from langchain.utilities.golden_query import GoldenQueryAPIWrapper
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from pathlib import Path
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.callbacks import get_openai_callback
import asyncio
from langchain.tools import PubmedQueryRun
from langchain.tools.python.tool import PythonREPLTool
from PIL import Image
from langchain.memory import ConversationBufferMemory
from unstructured import partition
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredImageLoader
import psycopg2
from psycopg2 import OperationalError

 

load_dotenv()

# Database connection
def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("PGVECTOR_DATABASE"),
        user=os.getenv("PGVECTOR_USER"),
        password=os.getenv("PGVECTOR_PASSWORD"), 
        host=os.getenv("PGVECTOR_HOST")
    )

connect_db()


CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER"),
    host=os.environ.get("PGVECTOR_HOST"),
    port=int(os.environ.get("PGVECTOR_PORT",)),
    database=os.environ.get("PGVECTOR_DATABASE"),
    user=os.environ.get("PGVECTOR_USER"),
    password=os.environ.get("PGVECTOR_PASSWORD")
)
COLLECTION_NAME = "documentation"

loader = DirectoryLoader(Path(os.getenv("DOCUMENTS_PATH")))
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

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

db.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
)
posgresVector = create_retriever_tool(
    db.as_retriever(), 
    "search_postgres",
    "Searches and returns documents from the postgreSQL vector database."
)


search = SerpAPIWrapper()
yfin =  YahooFinanceNewsTool()
wolfram = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()
golden_query = GoldenQueryAPIWrapper()
pubmed = PubmedQueryRun()
python_repl = PythonREPLTool()


llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo", streaming=True)

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Wolfram",
        func=wolfram.run,
        description="useful for when you need to answer questions about math"
    ),
    Tool(
        name="Arxiv",
        func=arxiv.run,
        description="useful for when you need to answer questions about scholarly articles"
    ),
    Tool(
        name="Golden Query",
        func=golden_query.run,
        description="useful for when you need to answer questions about business, finance, and natural language APIs"
    ),
    Tool(
        name="Postgres Vector",
        func=posgresVector.run,
        description="useful for when you need to answer questions about documents on the postgres database"
    ),
    Tool(
        name="Pubmed",
        func=pubmed.run,
        description="useful for when you need to query citations from pubmed about medical literature"
    ),
    Tool(
        name="Python REPL",
        func=python_repl.run,
        description="useful for when you need to write or execute python code"
        
    ),
    Tool(
        name="Yahoo Finance News",
        func=yfin.run,
        description="useful for when you need to query current finance information or make predictions on businesses"
        
    )
    
    
    
]

history = PostgresChatMessageHistory(
    connection_string=os.getenv("CONNECTION_STRING"),
    session_id="16390",
)

logo = "assets/logos/cropped_logo_blue.png"
st.sidebar.image(logo, use_column_width=True)

user_model = st.sidebar.selectbox("Select your GPT model", [
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo",
    "gpt-4"

])

user_temperature = st.sidebar.slider("Select your GPT temperature", 0.0, 1.0, 0.7, 0.01)
# Create a slider for tokens
selected_tokens = st.sidebar.slider("Select number of tokens", min_value=4000, max_value=16000, value=6000, step=10)

model = ChatOpenAI(temperature=user_temperature, model_name=user_model, max_tokens=selected_tokens, streaming=True,)

planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = initialize_agent(tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    

if "shared" not in st.session_state:
   st.session_state["shared"] = True

st.title("miniAGI 🤖")




if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    history.add_user_message(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        with get_openai_callback():
            response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
        history.add_ai_message(response)
        