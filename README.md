![image](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white) | ![image]([https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue))


# miniAGI

miniAGI is a Streamlit application designed to provide a chat interface with AGI (Artificial General Intelligence) capabilities. It leverages various toolkits to answer user queries with decision-making based on the plan and execution model from the Langchain framework.

## AI Features

- **Chat Interface**: Interactive chat interface where the plan and execute agent can choose between predefined functions. The prompt can also be a chain if you define your own chain prompt in the code. This will be among the first features addressed on the roadmap.
- **PostgreSQL**: Utilizes a PostgreSQL database for managing chat message history. Requires installation of PostgreSQL 15 and PGVector
- **PGVector**: Vector search for all filetypes storeable by PostgreSQL
- **Mathematical Queries**: Integration with Wolfram Alpha for answering mathematical questions.
- **Search Integration**: Ability to answer questions about current events using the SerpAPI.
- **Discord Chat Agent**: Import your discord chat data and have a conversation with it. 

## ML Features

- **Deeplake**: Large scale data hub for machine learning models and datasets
- **Banana**: Cloud provider for machine learning models
- **Potassium**: Banana offers a local server for development of machine learning models that will be uploaded to the cloud. similar to Django or Flask Server for ML models. This can be configured as your BANANA config values in the .env

## Quick Install


This guide will serve as a 'quick install':

## Installation

1. Clone the repository:

```bash
   git clone https://github.com/tdolan21/miniAGI.git
   cd miniAGI
   ```

## Requirements

The reccommended enviornment to run this application is through a conda virtual enviornment. 

If you are unfamillar with conda here are the install instructions for both [Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) and [Windows](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)


```bash 
conda create --name miniAGI python=3.9
conda activate miniAGI
```
If you want to use an nVidia GPU as compute for rendering machine learning models in the various modules, you should initialize the conda env like this before using pip install.

Ensure that the enviornment is activated before continuing this portion.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

If you just intend on using CPU compute, then a simple pip install will be enough once the conda enviornment has been created.

```bash
pip install -r "requirements.txt"
```
```bash
streamlit run app.py
```
