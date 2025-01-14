from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

## Till here, just checking that everything is working fine.

## LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot With OPENAI"

## Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, engine, temperature, max_token):
    llm = OllamaLLM(model=engine)
    output_parser = StrOutputParser()
    chain=prompt|llm|output_parser

    answer = chain.invoke({'question':question})
    return answer

## Setting the title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")

## Sidebar for setting
st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

## Drop down to select various OpenAI models
engine = st.sidebar.selectbox("Select an OpenAI model", ["gemma2:2b", "llama3.2", "mistral"])  ## we can add models later also based on the availability

## Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)
## 'value' is for the default value

## Main interface for user input
st.write("Go ahead and ask any Question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")

