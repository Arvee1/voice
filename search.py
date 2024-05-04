from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import streamlit as st

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

result = tool.run("Obama's first name?")
st.write(result)
