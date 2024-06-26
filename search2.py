from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import streamlit as st
import speech_recognition as sr
from langchain_community.llms import Replicate
import pyaudio
import wave
from audiorecorder import audiorecorder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# @st.cache_resource
# def call_conversation(prompt):
    # return conversation({"question": prompt})

# @st.cache_resource
# def get_messages():
    # if "messages" not in st.session_state:
        # st.session_state.messages = []
    # return st.session_state.messages

# st.session_state.messages = get_messages()
# for message in st.session_state.messages:
    # with st.chat_message(message["role"]):
        # st.markdown(message["content"])

# @st.cache_resource
# def setup_memory():
    # return ConversationBufferMemory(return_messages=True)

# memory = setup_memory()

# memory = ConversationBufferMemory(return_messages=True)

search = GoogleSearchAPIWrapper()

if "meta" not in st.session_state:
    st.write("Create Replicate LLM")
    llm = Replicate(
        model="meta/meta-llama-3-8b-instruct",
        model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
    )
    
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory = ConversationBufferMemory(llm=llm),    
    )

# st.write(st.session_state)

if "messages" not in st.session_state:
    st.session_state.messages = []

# st.write(st.session_state.conversation)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# @st.cache_resource
# def prompt_template():
    # return ChatPromptTemplate(
        # messages=[
            # SystemMessagePromptTemplate.from_template(
                # "You are a nice chatbot having a conversation with a human."
            # ),
            # The `variable_name` here is what must align with memory
            # MessagesPlaceholder(variable_name="chat_history"),
            # HumanMessagePromptTemplate.from_template("{question}")
            # ChatPromptTemplate.from_messages(
                # [
                    # ("system", "You are a helpful AI bot."),
                    # ("human", "{question}"),
                # ]
            # )
        # ]
    # )

# prompt = prompt_template()

# prompt = ChatPromptTemplate(
    # messages=[
        # SystemMessagePromptTemplate.from_template(
            # "You are a nice chatbot having a conversation with a human."
        # ),
        # The `variable_name` here is what must align with memory
        # MessagesPlaceholder(variable_name="chat_history"),
        # HumanMessagePromptTemplate.from_template("{question}")
        # ChatPromptTemplate.from_messages(
            # [
                # ("system", "You are a helpful AI bot."),
                # ("human", "{question}"),
            # ]
        # )
    # ]
# )
# result_ai = ""

# @st.cache_resource
# def setup_memory_buffer():
    # return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# memory = setup_memory_buffer()

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# @st.cache_resource
# def conv_chain():
    # return LLMChain(
        # llm=llm,
        # prompt=prompt,
        # verbose=False,
        # memory=memory
    # )
# conversation = conv_chain()

# conversation = LLMChain(
    # llm=llm,
    # prompt=prompt,
    # verbose=False,
    # memory=memory
# )

# llm = Replicate(
    # model="meta/meta-llama-3-70b-instruct",
    # model_kwargs={
        # "temperature": 0.75, 
        # "max_length": 500,
        # "max_tokens": 512,
        # "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        # "top_p": 1,
        # "memory": memory,
    # },
# )

# template = """You are a nice chatbot having a conversation with a human.

# Previous conversation:
# {chat_history}

# New human question: {question}
# Response:"""

# template = """You are a nice chatbot having a conversation with a human."""
# prompt = PromptTemplate.from_template(template)

# conversation = LLMChain(
    # llm=llm,
    # prompt=prompt,
    # verbose=True,
    # memory=memory
# )

def top5_results(query):
    return search.results(query, 5)

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    # func=top5_results,
    func=search.run,
)

st.title("👨‍💻 Wazzup!!!! I am Arvee's Personal Assistant?")
prompt = st.text_area("Please enter what you want to know.")

if st.button("Submit to AI", type="primary"):
    
    # result = tool.run(prompt)
    # result_ai = ""
    # for event in replicate.stream(
        # "meta/meta-llama-3-70b-instruct",
        # input={
            # "top_k": 50,
            # "top_p": 0.9,
            # "prompt": "Prompt: " + result,
            # "max_tokens": 512,
            # "min_tokens": 0,
            # "temperature": 0.6,
            # "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            # "presence_penalty": 1.15,
            # "frequency_penalty": 0.2
        # },
    # ):
    #run the model here
    # memory.chat_memory.add_user_message("Prompt: " + prompt)
    # for event in llm("Prompt: " + prompt):
        # result_ai = result_ai + (str(event))

    # Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
    # result_ai = conversation({"question": + prompt + ", " + result_ai})

    # result_ai = conversation({"question": prompt})
    
    # this is the orig run to uncomment
    # result_ai = llm("Prompt: " + prompt + ", " + result)
    
    # response_ai = conversation({"question": prompt + ", " + result})
    
    # st.write("Question and Search Result: " + prompt + " , " + result)
    # response_ai = conversation({"question": prompt})
    # response_ai = call_conversation(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm_response = st.session_state.conversation.run(
            prompt,
        )
        st.markdown(llm_response)
    st.session_state.messages.append({"role": "assistant", "content": llm_response})

    # st.write(llm_response)
    
    # print(response_ai)
    # json.loads()
    # print(response_ai['content'])
    # Assuming the 'chat_history' contains objects with a 'content' attribute
    # for i, message in enumerate(response_ai['chat_history']):
    #     print(f"Message {i + 1}: {message.content}")

    # print(f"AI Response: {response_ai['text']}")

    # result_ai = LLMChain(
        # llm=llm,
        # prompt="Prompt: " + prompt + " " + result_ai,
        # verbose=True,
        # memory=memory,
    # )
    
    # st.write(result_ai)
    # st.write(response_ai)
    
    # st.write(f"AI Response: {response_ai['text']}")

    # st.session_state.messages.append({"role": "user", "content": prompt})
    # response = f"Echo: {prompt}"

    # with st.chat_message("assistant"):
        # st.markdown(response)

    # st.session_state.messages.append({"role": "assistant", "content": response})

    # memory.chat_memory.add_ai_message(result_ai)

# This is the part where you can verbally ask about stuff
audio = audiorecorder("Click to record", "Click to stop recording")
      
if len(audio) > 0:
     # To play audio in frontend:
     st.audio(audio.export().read())  
     
     # To save audio to a file, use pydub export method:
     audio.export("audio.wav", format="wav")
     # To get audio properties, use pydub AudioSegment properties:
     st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
     
     soundfile = open("audio.wav", "rb")
     text = replicate.run(
          "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
          input={
            "task": "transcribe",
            "audio": soundfile,
            "language": "None",
            "timestamp": "chunk",
            "batch_size": 64,
            "diarise_audio": False
          }
     )
     st.write("what you said: " + text['text'])
     prompt = text['text']

     result_voice = tool.run(prompt)
     # st.write(augment_query)
     # st.write("###AI Response###")
     # The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.
     result_ai = ""
     # The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.
     for event in replicate.stream(
         "meta/meta-llama-3-70b-instruct",
         input={
             "top_k": 50,
             "top_p": 0.9,
             "prompt": "Prompt: " + result_voice,
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

     # output = replicate.run(
         # "afiaka87/tortoise-tts:e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71",
         # input={
             # "seed": 0,
             # "text": result_ai,
             # "preset": "fast",
             # "voice_a": "custom_voice",
             # "voice_b": "disabled",
             # "voice_c": "disabled",
             # "cvvp_amount": 0,
             # "custom_voice": "https://replicate.delivery/mgxm/671f3086-382f-4850-be82-db853e5f05a8/nixon.mp3"
         # }
     # )
     # print(output)
