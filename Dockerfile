# Use an Ubuntu image as a parent image
FROM ubuntu:20.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive

# Install some basic utilities and Python
RUN apt-get update && \
    apt-get install -y wget curl python3.9 python3-pip

RUN apt-get update && \
    apt-get install -y git
RUN apt-get update && \
    apt-get install -y libpq-dev

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Create and activate a new conda environment with Python 3.9
RUN conda create --name miniAGI python=3.9 -y
ENV PATH="/opt/conda/envs/miniAGI/bin:${PATH}"

# Install other conda and pip packages
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Set the working directory and copy the app and requirements file into it
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install



# Copy the app
COPY . /app

# Run the app
CMD ["streamlit", "run", "app.py"]
