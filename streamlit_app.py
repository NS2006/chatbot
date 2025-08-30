import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ---- Streamlit Setup ---- #
st.set_page_config(layout="wide")
st.title("Chatbot")

# ---- Sidebar Inputs ---- #
st.sidebar.header("Settings")

# Dropdown for model selection
model_options = ["llama3.2", "deepseek-r1:8b"]
MODEL = st.sidebar.selectbox("Choose a Model", model_options, index=0)

# Inputs for generation parameters
TEMPERATURE = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="Controls the randomness of the output. Lower values are more deterministic.")
TOP_P = st.sidebar.slider("Top P:", min_value=0.0, max_value=1.0, value=0.9, step=0.1, help="Nucleus sampling. The model considers tokens whose cumulative probability exceeds this value.")
MAX_SEQ_LEN = st.sidebar.slider("Max Sequence Length:", min_value=64, max_value=4096, value=1024, step=128, help="The maximum number of tokens to generate in the response.")

# ---- Function to Clear Memory When Settings Change ---- #
def clear_memory():
    """Clears the chat history and memory."""
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.success("Chat history and memory have been cleared due to a settings change.")

# Clear memory if any relevant settings are changed
if "prev_settings" not in st.session_state:
    st.session_state.prev_settings = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_seq_len": MAX_SEQ_LEN
    }
else:
    current_settings = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_seq_len": MAX_SEQ_LEN
    }
    if current_settings != st.session_state.prev_settings:
        clear_memory()
        st.session_state.prev_settings = current_settings

# ---- Initialize Chat Memory ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# ---- LangChain LLM Setup ---- #
# Pass the new parameters to the ChatOllama instance
llm = ChatOllama(
    model=MODEL, 
    streaming=True,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    num_ctx=8192,
    max_new_tokens=MAX_SEQ_LEN
)

# ---- Prompt Template ---- #
prompt_template = PromptTemplate(
    input_variables=["history", "human_input"],
    template="{history}\nUser: {human_input}\nAssistant:"
)

chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.memory)

# ---- Display Chat History ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Trim Function (Removes Oldest Messages) ---- #
def trim_memory():
    """Trims the chat history to the specified MAX_HISTORY."""
    while len(st.session_state.chat_history) > 5 * 2:  # Each cycle has 2 messages (User + AI)
        st.session_state.chat_history.pop(0)  # Remove oldest User message
        if st.session_state.chat_history:
            st.session_state.chat_history.pop(0)  # Remove oldest AI response

# ---- Handle User Input ---- #
if prompt := st.chat_input("Say something"):
    # Show User Input Immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({"role": "user", "content": prompt})  # Store user input

    # Trim chat history before generating response
    trim_memory()

    # ---- Get AI Response (Streaming) ---- #
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        for chunk in chain.stream({"human_input": prompt}):
            if isinstance(chunk, dict) and "text" in chunk:
                text_chunk = chunk["text"]
                full_response += text_chunk
                response_container.markdown(full_response)

    # Store response in session_state
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Trim history after storing the response
    trim_memory()