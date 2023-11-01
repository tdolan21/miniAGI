
-- Create prompt_templates table
CREATE TABLE IF NOT EXISTS prompt_templates (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    template TEXT NOT NULL
);

-- Populate prompt_templates table
INSERT INTO prompt_templates (name, template) VALUES
('QA', '{question}'),
('Python Basics', 'Write a Python function that provides a solution to {question} following PEP8 guidelines'),
('Machine Learning Intro', 'Explain the concept of {question} in machine learning.');

-- Create chains table
CREATE TABLE IF NOT EXISTS chains (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    llm TEXT NOT NULL,
    config_data JSON NOT NULL
);

-- Populate chains table (omitting id)
INSERT INTO chains (name, llm, config_data) VALUES
('NLP Processing', 'GPT-3', '{"steps": ["Tokenization", "Embedding", "Model Inference", "Post-Processing"]}');

-- Additional SQL instructions for vector extension and operations
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS items (id bigserial PRIMARY KEY, embedding vector(3));
INSERT INTO items (embedding) VALUES (ARRAY[1,2,3]), (ARRAY[4,5,6]);





