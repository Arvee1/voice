"""Python file to serve as the frontend"""
import streamlit as st
__import__('pysqlite3')
import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules["pysqlite3"]
from streamlit_chat import message
from langchain_community.llms import Replicate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationEntityMemory
from langchain.memory import ConversationKGMemory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
from langchain.chains import RetrievalQAWithSourcesChain
from bs4 import BeautifulSoup
import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


llm = Replicate(
     model="meta/meta-llama-3-8b-instruct",
     model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
)

embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
vectorstore = Chroma(embedding_function=HuggingFaceEmbeddings(model_name=embeddings_model_name), persist_directory="./chroma_db_oai")

# Memory for Retriever
memory = ConversationSummaryBufferMemory(llm=llm, input_key='question', output_key='answer', return_messages=True)

search = GoogleSearchAPIWrapper()

# Retriever
web_research_retriever = WebResearchRetriever.from_llm(
vectorstore=vectorstore,
llm=llm,
search=search,
)

# Initialize question-answering chain with sources retrieval
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever)

def get_text():
    # input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    input_text = st.text_input("You: ", key="input")
    return input_text

# st.title("👨‍💻 Wazzup!!!! What do you want to know about the Australian Federal Budget 2024?")
user_input = get_text()

if user_input:     
     # Query the QA chain with the user input question
     result = qa_chain({"question": user_input})
     
     # Print out the results for the user query with both answer and source url that were used to generate the answer
     st.write(result["answer"])
     st.write(result["sources"])


