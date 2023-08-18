from langchain import VectorDBQA
from langchain.agents import AgentType, initialize_agent, load_tools, Tool, Agent
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
import os
from langchain.document_loaders import WebBaseLoader
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from typing import List, Tuple
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain
from langchain.memory import PostgresChatMessageHistory



load_dotenv()



search = SerpAPIWrapper()
wolfram = WolframAlphaAPIWrapper()


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
    
    
]

history = PostgresChatMessageHistory(
    connection_string=os.getenv("CONNECTION_STRING"),
    session_id="16390",
)


model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)


    

if "shared" not in st.session_state:
   st.session_state["shared"] = True

st.title("miniAGI :computer:")
st.subheader("AGI with more targeted toolkits with decision making based on the plan and execution model from langchain")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    history.add_user_message(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
        history.add_ai_message(response)
        