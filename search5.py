__import__('pysqlite3')
import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules["pysqlite3"]
from langchain.vectorstores import Chroma
import os
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain
import streamlit as st

# Load environment variables for API keys
# os.environ["OPENAI_API_KEY"] = "[INSERT YOUR OPENAI API KEY HERE]"
# os.environ["GOOGLE_CSE_ID"] = "[INSERT YOUR GOOGLE CSE ID HERE]"
# os.environ["GOOGLE_API_KEY"] = "[INSERT YOUR GOOGLE API KEY HERE]"


# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

# Setup a Vector Store for embeddings using Chroma DB
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")

# Initialize memory for the retriever
memory = ConversationSummaryBufferMemory(llm=llm, input_key='question', output_key='answer', return_messages=True)

# Search
# Initialize Google Search API for Web Search
search = GoogleSearchAPIWrapper()

# Setup a Retriever
web_research_retriever = WebResearchRetriever.from_llm(
vectorstore=vectorstore,
llm=llm,
search=search,
)

# Define the User Input
user_input_question = "How do Planes work?"

# Initialize question-answering chain with sources retrieval
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever)

# Query the QA chain with the user input question
# result = qa_chain({"question": user_input_question})

# Print out the results for the user query with both answer and source url that were used to generate the answer
# st.write(result["answer"])
# st.write(result["sources"])

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
