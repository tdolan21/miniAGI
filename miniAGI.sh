#!/bin/bash

# Header and Metadata
echo "MiniAGI Installer v0.0.1"
echo "Author: tdolan21"


# Prerequisites Check
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it first."
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install it first."
    exit 1
fi
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install it first."
    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install it first."
    exit 1
fi
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL is not installed. Please install it first."
    exit 1
fi

# Create Conda Environment
echo "Creating conda environment..."
conda create --name miniAGI python=3.9 -y

# Activate environment
echo "Activating conda environment..."
source activate miniAGI || conda activate miniAGI

# Save the current directory
original_dir=$(pwd)

# Check for nVidia GPU support
read -p "Do you want to use an nVidia GPU for computing? [y/N]: " gpu_support

echo "Installing packages for nVidia GPU support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
 


# Installation
echo "Installing Python Packages..."
pip install \
  banana_dev==5.0.2 \
  gradio_tools==0.0.9 \
  langchain==0.0.270 \
  langchain_experimental==0.0.10 \
  pandas==2.0.3 \
  python-dotenv==1.0.0 \
  streamlit==1.25.0 \
  deeplake \
  numpy \
  openai \
  psycopg2-binary \
  psycopg \
  wolframalpha \
  google-search-results \
  anthropic \
  transformers \
  playwright \
  beautifulsoup4 \
  google-api-python-client \
  pathlib
  

# Check if pip installation succeeded
if [ $? -ne 0 ]; then
    echo "Failed to install Python packages."
    exit 1
fi

# Install Playwright Browsers
echo "Installing browsers for Playwright..."
playwright install

# Check if playwright installation succeeded
if [ $? -ne 0 ]; then
    echo "Failed to install browsers for Playwright."
    exit 1
fi

# Post-Installation
echo "Setting up PostgreSQL for Debian based distributions..."

echo "Installing PostgreSQL for Linux..."
sudo apt update
sudo apt install -y postgresql postgresql-contrib
   
echo "Exiting as PostgreSQL is not installed."




cd ~

# Install pgvector extension
echo "Installing pgvector extension..."

sudo apt update
sudo apt install -y build-essential postgresql-server-dev-all

cd /tmp
git clone --branch v0.5.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Structure PostgreSQL database
echo "Setting up PostgreSQL database..."

# Using environment variables for PostgreSQL username and password
pg_new_username=${POSTGRES_USER:-postgres}
pg_new_password=${POSTGRES_PASSWORD:-}

# Create new databases
sudo -u postgres createdb miniAGI
sudo -u postgres createdb messages

# Grant privileges
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE miniAGI TO $pg_new_username;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE messages TO $pg_new_username;"

# Create tables
sudo -u postgres psql miniAGI -c "CREATE TABLE prompt_templates (id serial PRIMARY KEY, name TEXT, template TEXT);"
sudo -u postgres psql miniAGI -c "CREATE TABLE chains (id serial PRIMARY KEY, name TEXT, config_data JSONB);"


    # pgvector is not installed, proceed with installation
echo "Installing pgvector extension..."
    # Execute commands in miniAGI database
sudo -u postgres psql miniAGI -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql miniAGI -c "CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));"
sudo -u postgres psql miniAGI -c "INSERT INTO items (embedding) VALUES (ARRAY[1,2,3]), (ARRAY[4,5,6]);"
    # Query and print output
output=$(sudo -u postgres psql miniAGI -c "SELECT * FROM items ORDER BY embedding <-> ARRAY[3,1,2] LIMIT 5;")
echo "Query Output:"
echo "$output"


echo "PostgreSQL setup and verification complete."


# Finish
echo "Installation complete."

# Ask user if they want to run the application
read -p "Do you want to run the application now? [y/N]: " run_app

if [[ $run_app == 'y' || $run_app == 'Y' ]]; then
    # Change to miniAGI directory
    # Return to the original directory
    cd "$original_dir"
    # Run the application
    echo "Running the application..."
    streamlit run app.py
    # Return to the original directory
    
fi

# Finish
echo "Installation complete."