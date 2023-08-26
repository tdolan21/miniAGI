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

load_dotenv()

question = ""

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

repo_id = st.selectbox("Select a model",
                       ("tiiuae/falcon-7b", "Qwen/Qwen-7B" ))  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 250}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
          
st.title("Huggingface models :rocket:")
st.info("This is a search engine for several hugging face models. This section is completely free, but requires a large amount of GPU compute.  Please refer to the documentation if you have any questions.")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    question = prompt
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        st.write(llm_chain.run(question))   
        
           