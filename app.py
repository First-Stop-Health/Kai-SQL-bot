import base64
import openai
import re
import streamlit as st
import os
import sqlalchemy
import json
import requests
import pandas as pd

from snowflake.sqlalchemy import URL
from sqlalchemy import text
from cryptography.hazmat.primitives import serialization
from langchain.memory import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler


#from src.workspace_connection.workspace_connection import connect_to_snowflake
from prompts import  custom_gen_sql
from few_shot_examples import custom_tool_list


image_path = os.path.dirname(os.path.abspath(__file__))
st.image(image_path+'/static/keboola_logo.png', width=200)
home_title = "Kai SQL Bot"  # Replace with your title

# Display the title with the "Beta" label using HTML styling
st.markdown(f"""# {home_title} <span style="color:#2E9BF5; font-size:16px;">Beta</span>""", unsafe_allow_html=True)

# Initialize the chat messages history
openai.api_key = st.secrets.OPENAI_API_KEY
msgs = StreamlitChatMessageHistory(key="chat_messages")
memory = ConversationBufferMemory(chat_memory=msgs)


# Model selection for the chatbot
model_selection = st.sidebar.selectbox("Choose a model", ['gpt-3.5-turbo-16k', 'gpt-4'], help="Select the model you want to use for the chatbot.")
llm = ChatOpenAI(model=model_selection, temperature=0, streaming=True) if model_selection != 'default' else OpenAI(temperature=0, streaming=True)



def _load_private_key_from_secrets(cfg) -> dict:
    """
    Build Snowflake SQLAlchemy connect_args for key-pair (JWT) auth.
    Supports:
      - Secret-only: private_key_pem or private_key_pem_b64 (no passphrase needed for unencrypted keys)
      - File-based:  private_key_path (omit passphrase for unencrypted files)
    """
    connect_args = {"authenticator": "SNOWFLAKE_JWT"}

    # File-based?
    if "private_key_path" in cfg:
        connect_args["private_key_file"] = cfg["private_key_path"]
        # No private_key_file_pwd for unencrypted key
        return connect_args

    # Secret-only?
    if "private_key_pem" in cfg or "private_key_pem_b64" in cfg:
        if "private_key_pem" in cfg:
            pem = cfg["private_key_pem"]
        else:
            pem = base64.b64decode(cfg["private_key_pem_b64"]).decode("utf-8")

        # Unencrypted key => password=None
        key_obj = serialization.load_pem_private_key(
            data=pem.encode("utf-8"),
            password=None,
        )
        # Export to PKCS#8 PEM for the connector
        private_key_pem = key_obj.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        connect_args["private_key"] = private_key_pem
        return connect_args

    raise RuntimeError("Missing key-pair secrets: provide private_key_pem (or private_key_pem_b64) or private_key_path.")


def _build_snowflake_url(cfg) -> URL:
    return URL(
        account=cfg["account_identifier"],
        user=cfg["user"],
        database=cfg["database_name"],
        schema=cfg["schema_name"],
        warehouse=cfg.get("warehouse_name"),
        role=cfg.get("role_name") or cfg.get("user"),
    )


def initialize_connection():
    # Prefer Keboola/Streamlit secrets; fallback to env for local dev
    cfg = st.secrets if "account_identifier" in st.secrets else os.environ

    url = _build_snowflake_url(cfg)
    connect_args = _load_private_key_from_secrets(cfg)

    engine = sqlalchemy.create_engine(url, connect_args=connect_args)
    db = SQLDatabase(engine=engine)

    toolkit = SQLDatabaseToolkit(llm=llm, db=db)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=50,
        extra_tools=custom_tool_list,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        return_intermediate_steps=True
    )
    return agent_executor, url, connect_args



agent_executor, conn_url, conn_args = initialize_connection()  


# Create a dictionary to store feedback counts
feedback_counts = {"thumbs_up": 0, "thumbs_down": 0}

# Function to handle user feedback
def handle_feedback(feedback_type):
    feedback_counts[feedback_type] += 1


ai_intro = "Hello, I'm Kai, your AI SQL Bot. I'm here to assist you with SQL queries. What can I do for you?"

if len(msgs.messages) == 0:
    msgs.add_ai_message(ai_intro)
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
if prompt := st.chat_input():
    msgs.add_user_message(prompt)
    st.chat_message("user").write(prompt)
    st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True) 
    prompt_formatted = custom_gen_sql.format(context=prompt)
    try:
        response = agent_executor.run(input=prompt_formatted, callbacks=[st_callback], memory=memory)
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")


    msgs.add_ai_message(response)
    st.chat_message("Kai").write(response)


with st.container():
    # get the output of the last message from the agent 
    if len(msgs.messages) > 1:
        last_output_message = msgs.messages[-1].content    
    
        # function to extact the sql from the response and execute it   
        def execute_sql():
    sql_matches = re.findall(r"```sql\n(.*?)\n```", last_output_message, re.DOTALL)
    for sql in sql_matches:
        try:
            # Use the connection + text() pattern (SQLAlchemy 2.x)
            engine = sqlalchemy.create_engine(conn_url, connect_args=conn_args)
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            st.sidebar.write("Results")
            st.sidebar.dataframe(df)

        except Exception as e:
            st.sidebar.warning(f"Invalid Query: {e}")

        if re.findall(r"```sql\n(.*?)\n```", last_output_message, re.DOTALL):
            st.button("Execute SQL", on_click=execute_sql)     

        def clear_chat():
            msgs.clear()
            
        st.sidebar.button("Clear Chat", on_click=clear_chat)

        # Create two columns with custom widths
        col_1, col_2 = st.columns([1, 5])
        # Apply custom CSS to reduce margin between columns
        st.markdown(
            """
            <style>
            .st-b3 {
                margin-left: -10px; /* Adjust the negative margin as needed */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # Display thumbs-up button in the first column
        with col_1:
            if st.button("👍"):
                handle_feedback("thumbs_up")
        # Display thumbs-down button in the second column
        with col_2:
            if st.button("👎"):
                handle_feedback("thumbs_down")
        if feedback_counts["thumbs_up"] > feedback_counts["thumbs_down"]:
            feedback = "positive"
        elif feedback_counts["thumbs_up"] < feedback_counts["thumbs_down"]:
            feedback = "negative"
        else:
            # If both counts are equal or both are 0, set a default feedback
            feedback = "neutral"

        #log_data = "User: " + last_user_message["content"] + "\n" + "Kai: " + last_output_message["content"] + "\n" + "feedback: " + feedback + "\n"
        headers = {'Content-Type': 'application/json'}
        # write the conversation back to keboola with feedback
        # check if the url exists in the secrets
        #if "url" in st.secrets:
            #r = requests.post(st.secrets["url"], data=log_data.encode('utf-8'), headers=headers)

