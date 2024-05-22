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
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# EMBED_MODEL = "all-MiniLM-L6-v2"
# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     # model_name=EMBED_MODEL
 # )

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "budget_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
 )

collection = client.get_or_create_collection(
     name=COLLECTION_NAME,
     embedding_function=embedding_func,
     metadata={"hnsw:space": "cosine"},
 )

# Load VectorDB
@st.cache_resource
def create_vector():
      with open("budget_ai.txt") as f:
          hansard = f.read()
          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=500,
              chunk_overlap=20,
              length_function=len,
              is_separator_regex=False,
          )
           
      texts = text_splitter.create_documents([hansard])
      documents = text_splitter.split_text(hansard)[:len(texts)]
     
      collection.add(
           documents=documents,
           ids=[f"id{i}" for i in range(len(documents))],
      )
      f.close()

create_vector()

search = GoogleSearchAPIWrapper()

def top5_results(query):
    return search.results(query, 5)

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    # func=top5_results,
    func=search.run,
)

@st.cache_resource
def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = Replicate(
            model="meta/meta-llama-3-8b-instruct",
            model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
        )
    # chain = ConversationChain(llm=llm)
    chain = ConversationChain(
            llm=llm,
            memory = ConversationBufferWindowMemory(k=2),
            # memory = ConversationEntityMemory(llm=llm),
            # memory = ConversationKGMemory(llm=llm)
            # memory = ConversationSummaryBufferMemory(llm=llm),
            # memory = ConversationBufferMemory(llm=llm),
        )
    return chain

chain = load_chain()
# Setup a Vector Store for embeddings using Chroma DB
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "search_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
 )

# vectorstore = client.get_or_create_collection(
     # name=COLLECTION_NAME,
     # embedding_function=embedding_func,
     # metadata={"hnsw:space": "cosine"},
 # )

# vectorstore = Chroma(embedding_function=embedding_func, persist_directory="./chroma_db_oai")
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
vectorstore = Chroma(embedding_function=HuggingFaceEmbeddings(model_name=embeddings_model_name), persist_directory="./chroma_db_oai")

# Setup a Retriever
llm = Replicate(
     model="meta/meta-llama-3-8b-instruct",
     model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
)
web_research_retriever = WebResearchRetriever.from_llm(
     vectorstore=vectorstore,
     llm=llm,
     search=search,
)

# Initialize question-answering chain with sources retrieval
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever)

# From here down is all the StreamLit UI.
# st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("👨‍💻 Wazzup!!!! Conversation with Arvee about the Budget 2024")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    # input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    input_text = st.text_input("You: ", key="input")
    return input_text

# st.title("👨‍💻 Wazzup!!!! What do you want to know about the Australian Federal Budget 2024?")
user_input = get_text()

if user_input:
    query_results = collection.query(
          query_texts=[user_input],
          # include=["documents", "embeddings"],
          include=["documents"],
          n_results=15,
    )
    augment_query = str(query_results["documents"])
    chain_input = user_input + ", " + augment_query
     
    user_input_question = "Who is the president of the United States?"
    # Query the QA chain with the user input question
    result = qa_chain({"question": user_input_question})
    # Print out the results for the user query with both answer and source url that were used to generate the answer
    # st.write(result["answer"])
    # st.write(result["sources"])

    # result = tool.run(user_input)
    # chain_prompt = user_input + " " + result
    # st.write(chain_prompt)
    # output = chain.run(input=user_input + ", " + result)
    # output = chain.run(input=chain_input)
    
    # st.session_state.past.append(user_input)
    # st.session_state.generated.append(output)
    # st.write(st.session_state)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
