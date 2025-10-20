import os
from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# load environment variables (if you store GROQ_API_KEY in .env)
load_dotenv()

# Default key from env (used if sidebar input is empty)
env_groq_key = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="Langchain — Chat with search", layout="wide")
st.title("Langchain — Chat with search")

st.write(
    "In this app we show the agent's thoughts/actions in the Streamlit UI using `StreamlitCallbackHandler`."
)

# Sidebar for settings
st.sidebar.title("Settings")
sidebar_key = st.sidebar.text_input("Enter your GROQ API key:", type="password", value="")

# Choose the key: use sidebar input if provided otherwise fallback to env
groq_api_key = sidebar_key.strip() or env_groq_key

if not groq_api_key:
    st.sidebar.warning("No GROQ API key provided — the LLM may fail without a valid key.")

# --- Setup external tool wrappers (ArXiv, Wikipedia, DuckDuckGo) ---
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arx = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchRun(name="Search")

# --- Chat state initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi — I'm a chatbot that can search the web. How may I help you?"
        }
    ]

# Display past messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
prompt = st.chat_input(placeholder="Ask me about machine learning, recent papers, or anything...")

if prompt:
    # append user message (use the actual prompt string, not the literal word "prompt")
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # instantiate LLM with the selected key
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", streaming=True)

    tools = [search, arx, wiki]

    # initialize an agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True
    )

    # Run the agent with Streamlit callback to show internal thoughts
    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # Pass the plain text prompt to agent.run (not the whole messages list)
        try:
            response = search_agent.run(prompt, callbacks=[cb])
        except Exception as e:
            response = f"Agent failed: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.write(response)
