![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/-PostgreSQL-336791?logo=postgresql&logoColor=white)
![Conda](https://img.shields.io/badge/-Conda-44A833?logo=anaconda&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Langchain](https://img.shields.io/badge/-Langchain-3E8FC9?)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![GitHub Followers](https://img.shields.io/github/followers/tdolan21?label=Follow&style=social)
![GitHub Stars](https://img.shields.io/github/stars/tdolan21?label=Stars&style=social)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Table of Contents
- [miniAGI](#miniagi)
  - [Philosophy](#philosophy)
  - [AI Features](#ai-features)
  - [ML Features](#ml-features)
  - [Quick Install](#quick-install)
  - [Requirements](#requirements)
- [In-Depth Guide](#in-depth-guide)
  - [Environment](#enviornment)
  - [PostgreSQL and PGVector](#postgresql-and-pgvector)
  - [API-Keys](#api-keys)
    - [Base Application (Plan and Execute Agent) (**REQUIRED**)](#base-application-plan-and-execute-agent-required)
    - [Claude integration (Plan and Execute Agent) (Optional)](#claude-integration-plan-and-execute-agent-optional)
    - [HuggingfaceHub (Plan and Execute Agent) (Optional, but recommended)](#huggingfacehub-plan-and-execute-agent-optional-but-recommended)
    - [Banana/Potassium (Plan and Execute Agent) (Fully Optional)](#bananapotassium-plan-and-execute-agent-fully-optional)
    - [Deeplake (Euclidean Distance Similarity Search with Images) (Optional enterprise tool)](#deeplake-euclidean-distance-similarity-search-with-images-optional-enterprise-tool)
- [Contributing to miniAGI](#contributing-to-miniagi)
  - [Code Standards](#code-standards)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Submitting Changes](#submitting-changes)
  - [Code of Conduct](#code-of-conduct)
  - [Licensing](#licensing)
  - [Questions?](#questions)

   - 

# miniAGI

miniAGI is a Streamlit application designed to provide a chat interface with AGI (Artificial General Intelligence) capabilities. It leverages various toolkits to answer user queries with decision-making based on the plan and execution model from the Langchain framework.

## Philosophy

I recognize the immediate criticism surrounding the fact that AGI applications can be heavy token consumers. This one has the potential to be the same based on your intended use. However, the agent in this application is the plan and execute agent from langchain, and is configured to have the decision making process only last 3 steps. This leads to an increased thought process rather than allowing the agent to make the choices on its own. It does not have long term context in this manor, but it does have both message memory and vector memory combined with several powerful tools.

The toolkit included in miniAGI is geared towards a machine learning enviornemnt where you can acquire data, manipulate it, store it, and recall it using your preferred AI model. The tools are pre-configured for demos that are conducive to learning how to use machine learning models and can be easily configured for use with your own machine learning tools. The toolkit for deeplake is specially useful for this as they have over 250 datasets that can be used with the plan and execution agent. This gives you a head start to your project no matter the level. You can then query this dataset through vector search including images if you so choose. 

Once your machine learning model is complete and you have a rich dataset, you can upload it to Banana or Potassium server for a cloud or local deployment to chat with the model using the plan and exexution agent. 

These tools are separate and are not all used in the same context. This application is intended to represent a fully functional data acquisition platform that allows the end2end prodution of machine learning models.

Considering this, all functionality of this application is experimental and should be treated as such. I will be including a docker image soon if you do not wish to download all of the different required programs locally. They are all required to run the application, but an API key is required to use the different services. This is another reason why its better to only include the toolsets where they are needed rather than to allow the opportunity to give the agent too much freedom.

The results of a more targeted goal and toolset include a more predictable experience where you can focus on experimentation with the plan and execute agent at whatever level you wish, rather than having it eat all your tokens and cost more. The agents will get there, but having more realistic goals and use cases is very important along the way.

DISCLAIMER: This agent does **not** have the freedom to use shell commands or freely access files on your machine. The only files it has access to locally are the files you put in the documents folder and the subdirectories required for each tool. 

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


1. Clone the repository:

```bash
   git clone https://github.com/tdolan21/miniAGI.git
   cd miniAGI
   mv .env.example .env
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

# In-Depth Guide
There will be a docker image deployed soon. In the meantime, its best to start with the conda enviornment mentioned above. If you would like to run this program locally you need to complete the conda set up as well as the PostgreSQL.

## Environment

This application requires installation of python and pip. Once you have python and pip installed, you should follow the steps outlined in the [Quick Install](#quick-install). This wil leave you with a python virtual enviornment that is configured for this application and is not affecting your personal machine. The requirements for this project are rather robust, but provide a richer experience than not using these tools. I consider this trade-off to be worth it. You may not and thats okay too.

## PostgreSQL and PGVector

First, You need to install PostgreSQL for your platform. Here is the download link for [Windows](https://www.postgresql.org/download/windows/) and [Linux](https://www.postgresql.org/download/linux/). 

Its recommended to also install pgAdmin during the install for PostgreSQL 15 so you can verify each feature is working easily.

Once you have PostgreSQL and pgAdmin installed, open pgAdmin and follow the install instructions listed in the pgVector link. This extension is not formally supported, but is an incredible addition to PostgreSQL allowing concurrent storage of both chat history, and the theoretical "brain" for your agent. This allows the user to further fine tune machine learning models that may be lacking in specific areas. 

[PGVector](https://github.com/pgvector/pgvector) is an extension for PostgreSQL that allows users to perform vector search within the same database as their traditional data. If you can store it on PostgreSQL, you can perform vector search with it by placing whatever documents you would like into the documents folder. This will load them into the vector extension of your database, allowing you to perpetually store documents and grow your vector search databse over time.

My preferred use case for this feature is to load the documentation to your ideal tech stack documentation into the documents to create your own agent toolkit. This will vectorize the documents and allow you to query them as you please.

## API-Keys

This application requires the use of several different API keys to use all the available features. You do not have to have the API keys for the application to work. Adding API keys essentially unlocks different features available throughout the application. If you do not have the neccesary API keys you will see an error thrown where they are used. In the future some of these features will be released as independent applications because they may be helpful to people as more specialized tools rather than a part of experimentation.

These are the API keys required for each section and they are cumulative, meaning you cant skip the first section and many things are required throughout:

### Base Application (Plan and Execute Agent) (**REQUIRED**)
   - [OpenAI](https://platform.openai.com/playground)
   - [Wolfram](https://products.wolframalpha.com/api/)
   - [SerpAPI](https://serpapi.com/) 

### Claude integration (Plan and Execute Agent) (Optional)

   Here is some more information on [Claude](https://www.anthropic.com/index/claude-2). Claude allows for longer context input for large datasets and very high level natural language processing. MiniAGI combines that with the ability to use other tools, store your results locally, and chain together more complex queries at once rather than needing to use one query at a time. MiniAGI has the potential to triple your productivity with Claude depending on your specific usecase, your mileage may vary.
   
   - [Anthropic API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
     
### HuggingfaceHub (Plan and Execute Agent) (Optional, but recommended)

   - [HuggingfaceHub API Token](https://huggingface.co/settings/profile): This API key allows you to configure whatever model you like to use locally through the transformes module. This API connects the 1000+ public models on Huggingface to a plan and execute chat enviornemnt with a plan and execution enviornemnt with file based vector search. This greatly expands the potential of these models. However, as of now these agents are largely not capable of utilizing other tools. Resulting in a more local experience with this agent to offer the best results.
     

### Banana/Potassium (Plan and Execute Agent) (Fully Optional)
   - [Banana](https://www.banana.dev/): This service allows the user to host their own machine learning models from the cloud and use them as an API in applications. It is integrated to this application in the same way as the other serivces, being the plan and execution agent. This service is costly and is a production machine learning enviornment. This is not required at all, but is very useful if you are building your own chat model or another model where you want to be able to use metrics in your other research. The default model that is configured is TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GPTQ, but you will have to configure and deploy whatever model you wish. The WizardVicuna ideaology was originally created by Eric Hartford, but on this project, the quantized version from TheBloke is used.
     
   - [Potassium](https://github.com/bananaml/potassium): If you are still working on your project you can connect this via the Potassium server functionality and then connect your local integration rather than a cloud integration. This allows the user to save money on deployment costs while still maintaining a similar testing experience.
   
### Deeplake (Euclidean Distance Similarity Search with Images) (Optional enterprise tool)

   - [Deeplake](https://www.deeplake.ai/): This is the most powerful of all the tool integrations by a large margin. Deeplake allows the user to connect the existing functionalities to a deeplake database that can be as large as you wish. They have many machine learning datasets available to the user through the hub functionality. In this application the configuration includes an API key and a hub dataset. This dataset can be public or that of your own creation. This feature is currently configured for image similarity search with a demo dataset hosted on my hub.
   - If you want to get the most out of this feature, you need to configure it for your own use cases. Whether that means hosting your own dataset on Activeloop Hub or that means utilizing one of the datasets on the activeloop [hub](https://datasets.activeloop.ai/docs/ml/datasets/).
   - There is almost certainly something on the activeloop public resources that is of use to you and can better aid your development process. They also offer a web based visalization tool that pairs quite well with the chat interface for further customization and tuning of your machine learning environment. 
   - This tool is currently under construction and is the most complex of all the integrations. However, it is functional at the base level if your configurations are correct. I will be updating the readme to fit whatever new features are added in the future, but expect this feautre to grow quite a bit.

# Contributing to miniAGI

## Code Standards
- Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
- Use descriptive variable names and comment your code.

## Testing
- Include unit tests for new functionality.
- Ensure all tests pass before submitting a pull request.

## Documentation
- Document all public functions and classes.
- Include examples where applicable.

## Submitting Changes
- Fork the repository and create a new branch for your feature or bug fix.
- Submit a pull request for review.
- Address any feedback from reviewers.

## Code of Conduct
- By using this application you accept the terms in our full CODE_OF_CONDUCT.md

## Licensing
- All contributions will be released under the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Questions?
- Join our [Discord](https://discord.gg/DTghN5YK7Y) or open an issue on GitHub.

We look forward to your contributions!

