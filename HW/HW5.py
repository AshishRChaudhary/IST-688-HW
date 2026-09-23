import json
import sys
from pathlib import Path

import chromadb
import streamlit as st
from openai import OpenAI

# Streamlit Community Cloud ships an sqlite3 that is too old for ChromaDB, so swap in
# pysqlite3 where it exists. It is neither needed nor installable on Windows.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

APP_DIR = Path(__file__).parent
REPO_DIR = APP_DIR.parent
CHROMA_PATH = str(REPO_DIR / "ChromaDB_for_HW4")
COLLECTION_NAME = "HW4Collection"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
BUFFER_TURNS = 5  # short-term memory: the last 5 user/assistant interactions

SYSTEM_PROMPT = """You are a student organization assistant for Syracuse University.

You have a relevant_club_info tool that searches a database of Syracuse student
organization pages. Call it whenever the question is about clubs, organizations, or
campus activities, and search with the words that describe what the student is looking
for rather than repeating their whole sentence.

Always make the source of your answer clear:
- If you used what the tool returned, say so and name the organizations you drew from.
- If the tool returned nothing useful and you answered from your own general knowledge
  instead, say that explicitly.
"""

CLUB_TOOL = {
    "type": "function",
    "function": {
        "name": "relevant_club_info",
        "description": "Search Syracuse University student organization pages and "
        "return the ones most relevant to a query, with their descriptions and "
        "contact details.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, e.g. 'robotics' or "
                    "'clubs for international students'.",
                }
            },
            "required": ["query"],
        },
    },
}

st.title("HW5: Club Chatbot with a Search Tool")

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# The collection built in HW4 is reused as-is; this page never rebuilds it.
if "HW5_VectorDB" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    st.session_state.HW5_VectorDB = chroma_client.get_or_create_collection(COLLECTION_NAME)

if st.session_state.HW5_VectorDB.count() == 0:
    st.error("The vector database is empty. Open HW 4 first to build it.")
    st.stop()


def relevant_club_info(query, n_results=5):
    """The tool the LLM calls: embed its query and return the closest club pages."""
    client = st.session_state.openai_client
    response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    results = st.session_state.HW5_VectorDB.query(
        query_embeddings=[response.data[0].embedding],
        n_results=n_results,
    )
    return [
        {"organization": metadata["organization"], "page_text": document}
        for metadata, document in zip(results["metadatas"][0], results["documents"][0])
    ]


def answer(messages_to_send, user_prompt):
    """Let the model search first, then answer with what the search returned."""
    client = st.session_state.openai_client

    # First call: the model decides whether to search, and picks the search words.
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages_to_send,
        tools=[CLUB_TOOL],
        tool_choice="auto",
    )
    message = response.choices[0].message

    if not message.tool_calls:
        st.write(message.content)
        return message.content

    messages_to_send.append(message.to_dict())
    for tool_call in message.tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        # A search request without a query falls back to the question as asked.
        query = arguments.get("query") or user_prompt
        clubs = relevant_club_info(query)
        st.caption(f"Searched the club database for: {query}")
        messages_to_send.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": json.dumps(clubs),
            }
        )

    # Second call, now that the model can see what it searched for. No tools are
    # passed, so it has to answer from those results instead of searching again.
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages_to_send,
        stream=True,
    )
    return st.write_stream(stream)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about Syracuse student organizations."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about a student organization"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    buffered_messages = st.session_state.messages[-(BUFFER_TURNS * 2 + 1):]
    messages_to_send = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + buffered_messages

    with st.chat_message("assistant"):
        response = answer(messages_to_send, prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
