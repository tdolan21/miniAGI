from langchain.chat_models import ChatVertexAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
from google.cloud import aiplatform
import os
load_dotenv()




chat = ChatVertexAI(model_name="codechat-bison")

messages = [
    HumanMessage(
        content="How do I create a python function to identify all prime numbers?"
    )
]
output = chat(messages)

print(output)