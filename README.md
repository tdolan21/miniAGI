# miniAGI Chat Application

miniAGI is an interactive chat application built with Streamlit, allowing users to engage with different AI agents. The application supports two types of agents: "Plan and Execute" and "ReAct."

## Features

- **Agent Selection**: Users can select between two types of agents ("Plan and Execute" and "ReAct") using a radio button.
- **Interactive Chat**: Users can ask questions, and the selected agent will respond. The conversation is displayed in the main interface.
- **Console Output**: The application captures the console output generated during the agent's response and displays it in the interface.
- **Configuration Page**: A placeholder for future configurations and tool selections.

## Components

### Libraries and Tools

The application leverages several libraries and tools, including:

- **langchain**: For agent initialization, execution, and tools.
- **dotenv**: To load environment variables such as API keys.
- **Streamlit**: For the web interface and user interaction.

### Functions

- `configuration_page()`: Displays the configuration page where future settings and tools can be managed.
- `chat_page()`: Handles the chat interface, agent selection, user input, agent response, and console output display.
- `main()`: The main function that initializes the sidebar navigation and calls the appropriate page function based on user selection.

### Initialization

- **OpenAI Models**: Initialize the OpenAI models and tools required for the agents.
- **SerpAPIWrapper**: For search functionality.
- **LLMMathChain**: For mathematical calculations.
- **Agent Initialization**: Initialize the "Plan and Execute" agent with the appropriate planner, executor, and tools.

## Usage

To run the application, simply execute the script:
``` bash
pip install langchain langchain_experimental openai wikipedia 
```
```bash
streamlit run app.py
```
