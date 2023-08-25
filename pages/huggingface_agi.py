from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain import PromptTemplate, LLMChain
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()



repo_id = "google/flan-t5-xxl"



template = """Question: {prompt}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["prompt"])

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)


st.title("HuggingFace AGI :computer:")

repo_id = st.selectbox('Please select a model:',(
                  "google/flan-t5-xxl",
                  "databricks/dolly-v2-3b",
                  "Writer/camel-5b-hf",
                  "Salesforce/xgen-7b-8k-base",
                  "tiiuae/falcon-40b",
                  "TheBloke/Wizard-Vicuna-30B-Uncensored-GGML"
            ))

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())

            

            template = """Question: {prompt}

            Answer: Let's think step by step."""

            prompt = PromptTemplate(template=template, input_variables=["prompt"])
            
            st.write(llm_chain.run(prompt))