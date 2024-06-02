from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import streamlit as st
import speech_recognition as sr
import replicate
import pyaudio
import wave
from audiorecorder import audiorecorder

search = GoogleSearchAPIWrapper()

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
    result = tool.run(prompt)
    result_ai = ""
    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": "Prompt: " + prompt + ", " + result,
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

