import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage

load_dotenv()

# llm = ChatOpenAI(api_key=st.secrets["openai_api_key"])
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chat_history = []

# Context for question
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")

])

# For answer
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Generate retrievers
documents = TextLoader("./docs/faq.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=30,  chunk_overlap=0, separator=".")
chunks = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
)
retriever = vectorstore.as_retriever()

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

def generate_response(query):
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    response = rag_chain.invoke({
        "input": query, 
        "chat_history": chat_history
        })
    return response

def query(query: str):
    response = generate_response(query)
    chat_history.extend([HumanMessage(content=query), response["answer"]])
    return response["answer"]
    # return {"answer": 'OK, I will do your request and ' + query + ':)'}