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
CHROMA_PATH = str(REPO_DIR / "ChromaDB_for_HW7")
COLLECTION_NAME = "HW7Collection"
EMBEDDING_MODEL = "text-embedding-3-small"
BUFFER_TURNS = 5  # short-term memory: the last 5 user/assistant interactions

MODELS = {
    "gpt-5-mini (lower cost)": "gpt-5-mini",
    "gpt-4.1 (higher cost)": "gpt-4.1",
}

INTEREST_QUERY = (
    "lawsuit, litigation, regulatory investigation, antitrust, fine, penalty, "
    "settlement, data breach, fraud, bankruptcy, merger or acquisition, executive "
    "departure, government probe, subpoena, class action"
)

# Used AI to generate this prompt, then lightly edited it for clarity and conciseness.
SYSTEM_PROMPT = """You are a news monitoring assistant for a global law firm. You report
only on the articles in the firm's news database, which covers its clients for the week
of 30 July to 6 August 2024.

You have two tools:
- most_interesting_news: use it when asked what is interesting, important or notable,
  with no particular company or topic in mind.
- news_about: use it for a specific company or topic, and for the most interesting news
  about a named company.

Call a tool before answering any question about the news. Answer only from what the tool
returns, and say so if it returns nothing relevant.

When you rank stories, give a numbered list from most to least interesting, and for each
one say what happened and why it matters to a law firm watching that client: litigation,
regulatory or criminal exposure, financial or deal risk, leadership change, or
reputational damage. Something routine, such as a share-count filing or a product
announcement, is low interest, and you should say so rather than inflate it.

Always name the company and the date for each story, and include its URL.
"""

INTERESTING_TOOL = {
    "type": "function",
    "function": {
        "name": "most_interesting_news",
        "description": "Return the articles that carry the strongest legal, regulatory "
        "or financial risk signals, for ranking. Use when no specific company or topic "
        "is named.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

NEWS_ABOUT_TOOL = {
    "type": "function",
    "function": {
        "name": "news_about",
        "description": "Search the news database for articles about a company or topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The company or topic to search for, for example "
                    "'Apple' or 'antitrust'.",
                }
            },
            "required": ["query"],
        },
    },
}

st.title("HW7: News Information Bot")

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Built by build_news_db.py, so this page only loads it.
if "HW7_VectorDB" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    st.session_state.HW7_VectorDB = chroma_client.get_or_create_collection(COLLECTION_NAME)

if st.session_state.HW7_VectorDB.count() == 0:
    st.error("The news database is empty. Run build_news_db.py first.")
    st.stop()


def search_news(query, n_results):
    """Embed a query and return the closest articles with their metadata."""
    client = st.session_state.openai_client
    response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    results = st.session_state.HW7_VectorDB.query(
        query_embeddings=[response.data[0].embedding],
        n_results=n_results,
    )
    return [
        {
            "company": metadata["company_name"],
            "date": metadata["date"],
            "url": metadata["url"],
            "article": document,
        }
        for metadata, document in zip(results["metadatas"][0], results["documents"][0])
    ]


def most_interesting_news():
    """Candidates for ranking: the articles richest in legal and financial risk."""
    return search_news(INTEREST_QUERY, n_results=20)


def news_about(query):
    """Articles about one company or topic."""
    return search_news(query, n_results=10)


def run_tool(name, arguments, user_prompt):
    if name == "most_interesting_news":
        st.caption("Searched the news database for risk signals")
        return most_interesting_news()
    # A search request without a query falls back to the question as asked.
    query = arguments.get("query") or user_prompt
    st.caption(f"Searched the news database for: {query}")
    return news_about(query)


def answer(messages_to_send, model, user_prompt):
    """Let the model search first, then report on what the search returned."""
    client = st.session_state.openai_client

    # First call: the model decides which tool to use, and what to search for.
    response = client.chat.completions.create(
        model=model,
        messages=messages_to_send,
        tools=[INTERESTING_TOOL, NEWS_ABOUT_TOOL],
        tool_choice="auto",
    )
    message = response.choices[0].message

    if not message.tool_calls:
        st.write(message.content)
        return message.content

    messages_to_send.append(message.to_dict())
    for tool_call in message.tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        articles = run_tool(tool_call.function.name, arguments, user_prompt)
        messages_to_send.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": json.dumps(articles),
            }
        )

    # Second call, now that the model can see the articles. No tools are passed, so it
    # has to report from those results instead of searching again.
    stream = client.chat.completions.create(
        model=model,
        messages=messages_to_send,
        stream=True,
    )
    return st.write_stream(stream)


with st.sidebar:
    st.header("LLM")
    model_label = st.selectbox("Model", list(MODELS.keys()))
    model = MODELS[model_label]
    st.caption(f"{st.session_state.HW7_VectorDB.count()} articles in the database")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me about this week's news. Try: find the most interesting "
            "news, or find news about Apple.",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about the news"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    buffered_messages = st.session_state.messages[-(BUFFER_TURNS * 2 + 1):]
    messages_to_send = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + buffered_messages

    with st.chat_message("assistant"):
        response = answer(messages_to_send, model, prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
