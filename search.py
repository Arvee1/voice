from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import streamlit as st

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

st.title("👨‍💻 Wazzup!!!! I am Arvee's Personal Assistant?")
prompt = st.text_area("Please enter what you want to know.")

if st.button("Submit to AI", type="primary"):
    result = tool.run(prompt)
    result_ai = ""
         for event in replicate.stream(
             "meta/meta-llama-3-70b-instruct",
             input={
                 "top_k": 50,
                 "top_p": 0.9,
                 "prompt": "Prompt: " + result
                 "max_tokens": 512,
                 "min_tokens": 0,
                 "temperature": 0.6,
                 "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                 "presence_penalty": 1.15,
                 "frequency_penalty": 0.2
             },
         ):
             result_ai = result_ai + (str(event))
         st.write(result_ai)
