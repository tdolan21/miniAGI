![miniAGI Logo](assets/logos/cropped_logo_blue.png)


# miniAGI

A Zero-shot ReAct agent that utilizes toolkits designed for data acquisition, manipulation, and processing. This environment contains everything you need to source, create, and deploy your machine learning models and chat with them using your configured agent.

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/-PostgreSQL-336791?logo=postgresql&logoColor=white)
![Conda](https://img.shields.io/badge/-Conda-44A833?logo=anaconda&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Langchain](https://img.shields.io/badge/-Langchain-3E8FC9?)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![GitHub Followers](https://img.shields.io/github/followers/tdolan21?label=Follow&style=social)
![GitHub Stars](https://img.shields.io/github/stars/tdolan21?label=Stars&style=social)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Issues](https://img.shields.io/github/issues/tdolan21/miniAGI.svg)](https://github.com/tdolan21/miniAGI/issues)
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/tdolan21)
[![Playwright](https://img.shields.io/badge/Playwright-v1.12.3-brightgreen)](https://github.com/microsoft/playwright)
[![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-v4.9.3-brightgreen)](https://www.crummy.com/software/BeautifulSoup/)
![Built with OpenAI](https://img.shields.io/badge/Built%20with-OpenAI-2877ff)
[![Join our Discord](https://dcbadge.vercel.app/api/server/rzfPeBnRpe)](https://discord.gg/rzfPeBnRpe)



# Table of Contents

1. [Overview](#Overview)
   - [Philosophy](#Philosophy)
   - [Disclaimer](#Disclaimer)
2. [Quick Install](#Quick-Install)
   - [Linux](#Linux)
   - [Docker](#Docker)
3. [Local Requirements](#Local-Requirements)
4. [In-Depth Guide](#In-Depth-Guide)
   - [Known Issues](#known-issues)
   - [Features](#Features)
   - [ML Features](#ML-Features)
   - [Environment](#Environment)
   - [PostgreSQL and PGVector](#PostgreSQL-and-PGVector)
   - [API-Keys](#API-Keys)
5. [Plugins](#Plugins)
   - [Claude integration](#Claude-integration)
   - [API Calls from external APIs](#API-Calls-from-external-APIs)
   - [Banana/Potassium](#Banana/Potassium)
   - [Deeplake](#Deeplake)
   - [Deeplake Codebase Agent](#Deeplake-Codebase-Agent)
6. [Contributing to miniAGI](#Contributing-to-miniAGI)
   - [Code Standards](#Code-Standards)
   - [Testing](#Testing)
   - [Documentation](#Documentation)
   - [Submitting Changes](#Submitting-Changes)
   - [Code of Conduct](#Code-of-Conduct)
7. [Licensing](#Licensing)
8. [Questions and Contact](#Questions)





# Overview

miniAGI is a Streamlit application designed to provide a chat interface with AGI (Artificial General Intelligence) capabilities. It leverages various toolkits to answer user queries with decision-making based on the plan and execution model from the Langchain framework.

Plugins are available at https://github.com/tdolan21/miniAGI-plugins

## Philosophy

I recognize the immediate criticism surrounding the fact that AGI applications can be heavy token consumers. This one has the potential to be the same based on your intended use. However, the agent in this application is the plan and execute agent from langchain, and is configured to have the decision making process only last 3 steps. This leads to an increased thought process rather than allowing the agent to make the choices on its own. It does not have long term context in this manor, but it does have both message memory and vector memory combined with several powerful tools.

The toolkit included in miniAGI is geared towards a machine learning environment where you can acquire data, manipulate it, store it, and recall it using your preferred AI model. The tools are pre-configured for demos that are conducive to learning how to use machine learning models and can be easily configured for use with your own machine learning tools. The toolkit for deeplake is specially useful for this as they have over 250 datasets that can be used with the plan and execution agent. This gives you a head start to your project no matter the level. You can then query this dataset through vector search including images if you so choose. 

Once your machine learning model is complete and you have a rich dataset, you can upload it to Banana or Potassium server for a cloud or local deployment to chat with the model using the plan and exexution agent. 

These tools are separate and are not all used in the same context. This application is intended to represent a fully functional data acquisition platform that allows the end2end prodution of machine learning models.

Considering this, all functionality of this application is experimental and should be treated as such. I will be including a docker image soon if you do not wish to download all of the different required programs locally. They are all required to run the application, but an API key is required to use the different services. This is another reason why its better to only include the toolsets where they are needed rather than to allow the opportunity to give the agent too much freedom.

The results of a more targeted goal and toolset include a more predictable experience where you can focus on experimentation with the plan and execute agent at whatever level you wish, rather than having it eat all your tokens and cost more. The agents will get there, but having more realistic goals and use cases is very important along the way.

### Disclaimer 
This agent does **not** have the freedom to use shell commands or freely access files on your machine. The only files it has access to locally are the files you put in the documents folder and the subdirectories required for each tool. 


## Quick Install

### Linux

This guide will serve as a 'quick install' for linux machines:


1. Clone the repository:

```bash
   git clone https://github.com/tdolan21/miniAGI.git
   cd miniAGI
   mv .env.example .env
   ```
   Once you have created you own .env file, you need to fill out the environment variables and run:

   ```bash
   bash miniAGI.sh
   ```

### Docker

**Windows machines MUST use docker**

The initial process is the same. Clone the repo, access the directory, create the .env.

 ```bash
 git clone https://github.com/tdolan21/miniAGI.git
 cd miniAGI
 mv .env.example .env
 ```
 After this is completed you will need to add your actual API credentials to the.env that has been created.

 #### Building the image

 This will build the image for the first time. It may take a few minutes. You have time to get more coffee.

 ```bash
 docker build -t miniagi .
 ```
 #### Running the image

 Replace "mysecretpassword" with your desired password.

 ```bash
 docker run -e "POSTGRES_PASSWORD=mysecretpassword" miniagi
 ```
 #### Using Docker Compose

 ```bash
 docker-compose up       # To start the services
 docker-compose down -v    # To stop the services
 ```


## Local Requirements

**This application does NOT work locally with windows**

More information can be found at https://github.com/pgvector/pgvector

This section outlines what the miniAGI.sh script is doing.

The reccommended environment to run this application is through a conda virtual environment. 

If you are unfamillar with conda here are the install instructions for [Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

```bash 
conda create --name miniAGI python=3.9
conda activate miniAGI
```
If you want to use an nVidia GPU as compute for rendering machine learning models in the various modules, you should initialize the conda env like this before using pip install.

Ensure that the environment is activated before continuing this portion.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

If you just intend on using CPU compute, then a simple pip install will be enough once the conda environment has been created.

```bash
pip install -r "requirements.txt"
```
```bash
streamlit run app.py
```

# In-Depth Guide

## Known issues

 You will get this error: 

 ```
 miniagi-db-1   | 2023-09-11 17:19:38.073 UTC [68] FATAL:  password authentication failed for user "postgres"
miniagi-db-1   | 2023-09-11 17:19:38.073 UTC [68] DETAIL:  Connection matched pg_hba.conf line 100: "host all all all scram-sha-256"
miniagi-db-1   | 2023-09-11 17:19:38.090 UTC [69] FATAL:  password authentication failed for user "postgres"
miniagi-db-1   | 2023-09-11 17:19:38.090 UTC [69] DETAIL:  Connection matched pg_hba.conf line 100: "host all all all scram-sha-256"
 ```

 This error is a bug with the usage of the pgvector docker image in conjunction with using standard postgreSQL functionality.

 However, even with the error present, database connection can be accurately proven through the different feautres and everything still works as expected.

 The database should not throw this error as the connection is made via either SHA-256 password or trust on local networks.

 If you wish to change this or attempt to fix the bug, the file you will need is the pg_hba.conf and is located at:

 ```
 /var/lib/postgresql/data
 ```

## Features

- **Chat Interface**: Interactive chat interface where the plan and execute agent can choose between predefined functions. The prompt can also be a chain if you define your own chain prompt in the code. This will be among the first features addressed on the roadmap.
- **PostgreSQL**: Utilizes a PostgreSQL database for managing chat message history. Requires installation of PostgreSQL 15 and PGVector
- **PGVector**: Vector search for all filetypes storeable by PostgreSQL
- **Mathematical Queries**: Integration with Wolfram Alpha for answering mathematical questions.
- **Search Integration**: Ability to answer questions about current events using the SerpAPI.
- **Deeplake**: This feature is available as a plugin because it is a more niche usecase. Activeloop provides 250+ datasets for machine learning and this application allows you to import them to chat with or use for image similarity search. The plugin is available [here](https://github.com/tdolan21/miniAGI-plugins)
- **Banana**: Plan and Execute agent integration for your own cloud hosted models through banana.
- **Potassium**: Banana offers a local server for development of machine learning models that will be uploaded to the cloud. similar to Django or Flask Server for ML models. This can be configured as your BANANA config values in the .env
- **Train and Visualize**: I included a rudimentary space to test and visualize small CSV datasets and train a model based on the data you select. It will then provide a small visuzalization on the actual model using shad. The feature is currently limited in functionality, but will be expanded upon greatly.
- **HuggingFace Hub**: The huggingface agi section allows the user to explore the zero-shot ReAct agent using their own resources and model of their choosing. Models can be saved to the database for future use if you find one you like. This section relies on your CPU/GPU and is quite resource intensive. 10GB+ VRAM is ideal for the provided 7B parameter models. CPU embeddings are completed by FAISS. 
- **Playwright Web Retriever**: Allows the agent to interact with a browser in a dedicated playwright environment. This is not a search tool useable by miniAGI, but an additional agent with a different purpose.



## Environment

This application requires installation of python and pip. Once you have python and pip installed, you should follow the steps outlined in the [Quick Install](#quick-install). This will leave you with a Python virtual environment that is configured for this application and is not affecting your personal machine. The requirements for this project are rather robust, but provide a richer experience than not using these tools. I consider this trade-off to be worth it. You may not and thats okay too.

## PostgreSQL and PGVector

First, You need to install PostgreSQL for your platform. Here is the download link for [Windows](https://www.postgresql.org/download/windows/) and [Linux](https://www.postgresql.org/download/linux/). 

Its recommended to also install pgAdmin during the install for PostgreSQL 15 so you can verify each feature is working easily.

Once you have PostgreSQL and pgAdmin installed, open pgAdmin and follow the install instructions listed in the pgVector link. This extension is not formally supported, but is an incredible addition to PostgreSQL allowing concurrent storage of both chat history, and the theoretical "brain" for your agent. This allows the user to further fine tune machine learning models that may be lacking in specific areas. 

[PGVector](https://github.com/pgvector/pgvector) is an extension for PostgreSQL that allows users to perform vector search within the same database as their traditional data. If you can store it on PostgreSQL, you can perform vector search with it by placing whatever documents you would like into the documents folder. This will load them into the vector extension of your database, allowing you to perpetually store documents and grow your vector search databse over time.

My preferred use case for this feature is to load the documentation to your ideal tech stack documentation into the documents to create your own agent toolkit. This will vectorize the documents and allow you to query them as you please.

## API-Keys

This application requires the use of several different API keys to use all the available features. You do not have to have the API keys for the application to work. Adding API keys essentially unlocks different features available throughout the application. If you do not have the neccesary API keys you will see an error thrown where they are used. In the future some of these features will be released as independent applications because they may be helpful to people as more specialized tools rather than a part of experimentation.

These are the API keys required for each section and they are cumulative, meaning you cant skip the first section and many things are required throughout:

### Base Application (Zero-shot ReAct Agent) (**REQUIRED**)
   - [OpenAI](https://platform.openai.com/playground): Needed for the entire application.
   - [Wolfram](https://products.wolframalpha.com/api/): Needed as a tool for miniAGI.
   - [SerpAPI](https://serpapi.com/): Needed for the search tool used by miniAGI.
   - [Golden](https://docs.golden.com/reference/getting-started): Golden is an API with technical data
   - [HuggingfaceHub API Token](https://huggingface.co/settings/profile): This API key allows you to configure whatever model you like to use locally through the transformes module. This API connects the 1000+ public models on Huggingface to a plan and execute chat enviornemnt with a plan and execution enviornemnt with file based vector search. This greatly expands the potential of these models. However, as of now these agents are largely not capable of utilizing other tools. Resulting in a more local experience with this agent to offer the best results.
   - [Deeplake](https://www.deeplake.ai/)

### Google Custom Search
   - [Google Custom Search](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwibq8LliPGAAxV5F1kFHdIgDB0QFnoECBAQAQ&url=https%3A%2F%2Fdevelopers.google.com%2Fcustom-search%2Fv1%2Fintroduction&usg=AOvVaw1pz8lxgN0a7s9q2JAKILDF&opi=89978449)
   - [Google Api Key](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiG66X0iPGAAxWkVDUKHRBQAzAQFnoECBUQAQ&url=https%3A%2F%2Fsupport.google.com%2Fgoogleapi%2Fanswer%2F6158862%3Fhl%3Den&usg=AOvVaw2B82tUsH0M884zBo23S1in&opi=89978449)
   
## Plugins

I have started a plugin repository named [miniAGI-plugins](https://github.com/tdolan21/miniAGI-plugins). This repo is currently under construction, but I have several plugins already functioning. I just need to standardize them and develop a structure for everyone to use.

These plugins include game simulations in a gymnasium environment and a debate simulation with an agent moderator. They will debate an ethical issue with a search and memory toolkit.

Other examples include the ability to chat with your discord data that can be retrieved via discord. The descriptions below are examples of the plugins currently available.

### Claude integration (Plan and Execute Agent) (Optional)

   Here is some more information on [Claude](https://www.anthropic.com/index/claude-2). Claude allows for longer context input for large datasets and very high level natural language processing. MiniAGI combines that with the ability to use other tools, store your results locally, and chain together more complex queries at once rather than needing to use one query at a time. MiniAGI has the potential to triple your productivity with Claude depending on your specific usecase, your mileage may vary.
   
   - [Anthropic API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)

### API Calls from external APIs

- [TMDB BEARER TOKEN](https://www.themoviedb.org/settings/api): In the section where the agent is able to make api calls to external apis, the first implemented API is from the movie database. This allows you to collect information on movies where the results are stored in YAML files and imported to your PGVector database for later use.
- [LISTEN_API_KEY](https://www.listennotes.com/api/): This api is used in the same way, but this api has access to ~3.2M podcasts and ~175M episodes of podcasts. It is updated regularly and can be used for data acquisition or simple research. The results of this are also stored in YAML files and can be utilized by PGVector. Example yaml output can be found in 'documents/api_yaml'
     

### Banana/Potassium (Plan and Execute Agent) (Fully Optional)
   - [Banana](https://www.banana.dev/): This service allows the user to host their own machine learning models from the cloud and use them as an API in applications. It is integrated to this application in the same way as the other serivces, being the plan and execution agent. This service is costly and is a production machine learning environment. This is not required at all, but is very useful if you are building your own chat model or another model where you want to be able to use metrics in your other research. The default model that is configured is TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GPTQ, but you will have to configure and deploy whatever model you wish. The WizardVicuna ideaology was originally created by Eric Hartford, but on this project, the quantized version from TheBloke is used.
     
   - [Potassium](https://github.com/bananaml/potassium): If you are still working on your project you can connect this via the Potassium server functionality and then connect your local integration rather than a cloud integration. This allows the user to save money on deployment costs while still maintaining a similar testing experience.
   
### Deeplake (Euclidean Distance Similarity Search with Images) (Optional enterprise tool)

   - [Deeplake](https://www.deeplake.ai/): This is the most powerful of all the tool integrations by a large margin. Deeplake allows the user to connect the existing functionalities to a deeplake database that can be as large as you wish. They have many machine learning datasets available to the user through the hub functionality. In this application the configuration includes an API key and a hub dataset. This dataset can be public or that of your own creation. This feature is currently configured for image similarity search with a demo dataset hosted on my hub.
   - If you want to get the most out of this feature, you need to configure it for your own use cases. Whether that means hosting your own dataset on Activeloop Hub or that means utilizing one of the datasets on the activeloop [hub](https://datasets.activeloop.ai/docs/ml/datasets/).
   - There is almost certainly something on the activeloop public resources that is of use to you and can better aid your development process. They also offer a web based visalization tool that pairs quite well with the chat interface for further customization and tuning of your machine learning environment. 
   - This tool is currently under construction and is the most complex of all the integrations. However, it is functional at the base level if your configurations are correct. I will be updating the readme to fit whatever new features are added in the future, but expect this feautre to grow quite a bit.

### Deeplake Codebase Agent

    - This agent uses The QARetrievalChain and the Deeplake Hub to create a vector search database out of your existing codebase. The user can define a location to create the vectorstore and it will discover all of the available python files in that folder structure and split them in half. It could be an odd number because it keeps the structure of the functions rather than a hard character limit. 


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

