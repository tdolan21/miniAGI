import streamlit as st
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import SerpAPIWrapper, LLMMathChain, Wikipedia
from langchain.agents.tools import Tool
from langchain.agents.react.base import DocstoreExplorer
from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from dotenv import load_dotenv
import os
import io
import sys


load_dotenv()

# Initialize Components
os.environ['OPENAI_API_KEY'] = "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"
serpapi_api_key = os.environ['SERPAPI_API_KEY'] = "e61b968b298d77f42856fe40e6f67c32d5bc1a2d69e74d312a2e12f0002130b5"

docstore = DocstoreExplorer(Wikipedia())
search = SerpAPIWrapper()
llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(name="Search", func=search.run, description="useful for when you need to answer questions about current events"),
    Tool(name="Calculator", func=llm_math_chain.run, description="useful for when you need to answer questions about math"),
]
model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)


def configuration_page():
    st.title("Configuration")
    st.sidebar.subheader('Tools Selection')
    

def chat_page():
    st.title('💬 Chat with miniAGI')

    # Redirect standard output to a custom StringIO object
    original_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Radio button to select agent type
    selected_agent = st.radio("Select Agent", ["Plan and Execute", "ReAct"])

    # Input field for user's question
    user_input = st.text_input("Ask a question:")

    # Button to trigger the agent's response
    if st.button('Ask'):
        # Add user input to display
        st.write(f"You: {user_input}")

        # Determine response based on selected agent
        if selected_agent == "Plan and Execute":
            response = agent.run(user_input)
        else:
            llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo")  
            tools = [
                Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search",
    ),
                Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup",
    ),
]           
            react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
            response = react.run(user_input)

        # Add agent response to display
        st.write(f"Agent: {response}")

        # Restore standard output
        sys.stdout = original_stdout

        # Get the console output from the custom StringIO object
        console_output = new_stdout.getvalue()

        # Add console output to display if there's any
        if console_output.strip():
            st.subheader("Console Output:")
            st.code(console_output.strip())

    # Clear the custom StringIO object for the next interaction
    new_stdout.truncate(0)
    new_stdout.seek(0)




def main():
    # Create sidebar menu
    page_selection = st.sidebar.radio("Navigate:", [ "Chat with miniAGI", "Configuration"])

    if page_selection == "Configuration":
        configuration_page()
    elif page_selection == "Chat with miniAGI":
        chat_page()
    

if __name__ == '__main__':
    main()

