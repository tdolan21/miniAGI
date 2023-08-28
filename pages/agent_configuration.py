import streamlit as st
from dotenv import load_dotenv
import streamlit as st
import psycopg2
import os
load_dotenv()

st.title("miniAGI agent configurations :computer:")
st.info("This is a configuration page for the miniAGI agent. If the prompt templates or chains are useable in your selected module, they will be in the sidebar.")

# Database connection
def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("PGVECTOR_DATABASE"),
        user=os.getenv("PGVECTOR_USER"),
        password=os.getenv("PGVECTOR_PASSWORD"), 
        host=os.getenv("PGVECTOR_HOST")
    )

# Save new template to database
def save_template(name, template):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO prompt_templates (name, template) VALUES (%s, %s) RETURNING id;", (name, template))
    id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return id

def fetch_templates():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, template FROM prompt_templates;")
    templates = cur.fetchall()
    cur.close()
    conn.close()
    return templates

prompt_templates, chains, other = st.tabs(["Prompt Templates", "Chains", "Other"])

# Prompt Templates configuration tab
with prompt_templates:
    
    # Display existing templates
    existing_templates = fetch_templates()
    st.write("Existing Templates:")
        
    selected_existing_template = st.selectbox("View existing templates", existing_templates, format_func=lambda x: x[1])
        
    if selected_existing_template:
        with st.container():
            st.write("Template Content:")
            st.write(selected_existing_template[2])

    st.divider()

    new_name = st.text_input("Enter the name for your new template", "")
    new_template = st.text_area("Enter your new template", "")
    if st.button("Save Template"):
        save_template(new_name, new_template)
        st.success("Template saved successfully.")


# Chains configuration tab
with chains:

    st.divider()
    st.write("Chains are not yet implemented.")


# Future Configurations implementation
with other:

    st.divider()
    st.write("Other configurations are not yet implemented.")
