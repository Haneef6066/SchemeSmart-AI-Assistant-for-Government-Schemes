import os
import time
import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from mistralai import Mistral
from pypdf import PdfReader
import speech_recognition as sr

# Streamlit page configuration
st.set_page_config(page_title="SchemeSmart", page_icon="flag.jpeg", layout="wide")

# Initialize session state attributes
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "text" not in st.session_state:
    st.session_state.text = ""

# Function to add messages to the chat
def add_message(msg, agent="ai", stream=True, store=True, avatar=None):
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent, avatar=avatar):
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)

    if store:
        st.session_state.messages.append(dict(agent=agent, content=output, avatar=avatar))

# Cache the client for reuse
@st.cache_resource
def get_client():
    api_key = "pULoY3kPGs8W1MDG8nFPMLouXVl6vL2V"
    if not api_key:
        st.error("The MISTRAL_API_KEY is not set. Please set it in the code.")
        st.stop()
    return Mistral(api_key=api_key)

CLIENT: Mistral = get_client()

# Prompt template for the Mistral model
PROMPT = """
An excerpt from a document is given below.

---------------------
{context}
---------------------

Given the document excerpt, answer the following query.
If the context does not provide enough information, decline to answer.
Do not output anything that can't be answered from the context.

Query: {query}
Answer:
"""

# Function to generate a reply based on a query
def reply(query: str, index: IndexFlatL2):
    embedding = embed(query)
    embedding = np.array([embedding])

    _, indexes = index.search(embedding, k=2)
    context = [st.session_state.chunks[i] for i in indexes.tolist()[0]]

    messages = [
        {"role": "user", "content": PROMPT.format(context='\n'.join(context), query=query)}
    ]

    response = CLIENT.chat.complete(model="mistral-tiny", messages=messages)
    add_message(response.choices[0].message.content, agent="ai", avatar="logo2.png")

# Function to build an index from a PDF file
def build_index(pdf_file_path):
    if not pdf_file_path:
        st.session_state.clear()
        return

    reader = PdfReader(pdf_file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n\n"

    st.session_state.text += text

    chunk_size = 2048
    chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

    if len(chunks) > 500:
        st.error("Document is too long!")
        st.session_state.clear()
        return

    st.sidebar.info(f"Indexing {len(chunks)} chunks.")
    progress = st.sidebar.progress(0)

    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(embed(chunk))
        progress.progress((i + 1) / len(chunks))

    embeddings = np.array(embeddings)

    st.session_state.chunks.extend(chunks)
    if st.session_state.index is None:
        dimension = embeddings.shape[1]
        st.session_state.index = IndexFlatL2(dimension)
    st.session_state.index.add(embeddings)

# Function to simulate streaming text character by character
def stream_str(s, speed=250):
    for c in s:
        yield c
        time.sleep(1 / speed)

# Cache the embedding function to reuse for queries
@st.cache_data
def embed(text: str):
    return CLIENT.embeddings.create(model="mistral-embed", inputs=[text]).data[0].embedding

# Clear the conversation when the button is pressed
if st.sidebar.button("🔴 Reset conversation"):
    st.session_state.messages = []

# Display chat messages with avatars
for message in st.session_state.messages:
    with st.chat_message(message["agent"], avatar=message.get("avatar")):
        st.write(message["content"])

# Initialize the PDF files
pdf_files_path = r"D:\SchemeSmart-main\SchemeSmart-main\database"
pdf_files = [os.path.join(pdf_files_path, f) for f in os.listdir(pdf_files_path) if f.endswith('.pdf')]

if not pdf_files:
    st.stop()

# Build the index for each PDF file
for pdf_file_path in pdf_files:
    build_index(pdf_file_path)

# Get the index from session state
index: IndexFlatL2 = st.session_state.index

# Add a voice input button
recognizer = sr.Recognizer()

if st.button("🎙️ Speak your query"):
    with sr.Microphone() as source:
        st.info("Listening... Please speak clearly.")
        try:
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            st.success(f"Your query: {query}")
            add_message(query, agent="human", stream=False, store=True)
            reply(query, index)
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand your voice. Please try again.")
        except sr.RequestError as e:
            st.error(f"Voice recognition error: {e}")

# Add an initial message if no conversation exists
if not st.session_state.messages:
    add_message("How may I help you today?", agent="ai", avatar="logo2.png")
