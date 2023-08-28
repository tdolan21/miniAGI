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
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.memory import PostgresChatMessageHistory
from langchain.utilities.arxiv import ArxivAPIWrapper
from langchain.utilities.golden_query import GoldenQueryAPIWrapper

load_dotenv()



search = SerpAPIWrapper()
wolfram = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()
golden_query = GoldenQueryAPIWrapper()


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
    )
    
    
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
        