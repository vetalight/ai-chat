import streamlit as st
# import pandas as pd
# import numpy as np
import openai
import os
from dotenv import load_dotenv
from query import query

# Title of the app
st.title('My first Streamlit app')

load_dotenv()

# API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = st.secrets["openai_api_key"]
openai.api_key = API_KEY

chat_placeholder = st.empty()


def init_chat_history():
    """Initialize chat history with a system message."""
    if "messages" not in st.session_state:
        # st.session_state["messages"] = []
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]


def start_chat():
    """Start the chatbot conversation."""
    # Display chat messages from history on app rerun
    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from Chat models
        response = query(prompt)
        print("response:--- ", response)
        
        # message_placeholder.markdown(response)
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    init_chat_history()
    start_chat()