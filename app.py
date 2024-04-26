import streamlit as st
import replicate
import os
from key import get_key
# App title
st.set_page_config(page_title="Maths LLM")


# Main content
st.title("Maths LLM")
st.write("Solve maths using DeepSeek-Math-7B-instruct LLM Model. ")
st.write("DeepSeekMath is initialized with DeepSeek-Coder-v1.5 7B and continues pre-training on math-related tokens sourced from Common Crawl, together with natural language and code data for 500B tokens. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. [Model paper link](\"https://arxiv.org/pdf/2402.03300.pdf\")")
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(logo.png);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

# Replicate Credentials
with st.sidebar:
    logo_url = './logo.png'
    st.sidebar.image(logo_url, width=100)
    
os.environ['REPLICATE_API_TOKEN'] = get_key()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Got any equations for me?", "top_k": 50,
        "top_p": 0.9,
        "temperature": 1,
        "max_new_tokens": 500}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Got any equations for me?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


def generate_llm_response(prompt_input):
    output = replicate.run(
    "deepseek-ai/deepseek-math-7b-instruct:8328993709e75f2e6417d9ac24a1330961545f6d05d1ab13cdfdd21c00cb1a6e",
    input={
        "text": f"Solve this algebraic equation and give me the proper workings:\"{prompt_input}\"",
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 1,
        "max_new_tokens": 500
    }
)
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not get_key()):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llm_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
