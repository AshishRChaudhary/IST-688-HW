import sys
from pathlib import Path

import chromadb
import streamlit as st
from bs4 import BeautifulSoup
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
DATA_FOLDER = REPO_DIR / "su_orgs"
CHROMA_PATH = str(REPO_DIR / "ChromaDB_for_HW4")
COLLECTION_NAME = "HW4Collection"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
EMBED_BATCH = 100  # embed in batches; one call per chunk would be ~1000 requests
BUFFER_TURNS = 5  # keep the last 5 user/assistant interactions

SYSTEM_PROMPT = """You are a student organization assistant for Syracuse University.

Each question arrives with excerpts from student organization pages as context. Use that
context to answer the question.

Always make the source of your answer clear:
- If you used the provided context, say so and name the organizations you drew from.
- If the context does not cover the question and you answered from your own general
  knowledge instead, say that explicitly.
"""

st.title("HW4: iSchool Chatbot Using RAG")

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def extract_text_from_html(html_path):
    """Pull the readable text out of a saved organization page."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    name = soup.title.string.strip() if soup.title and soup.title.string else html_path.stem
    return name, soup.get_text(separator=" ", strip=True)


def split_into_two(text):
    """Chunking method: split each page into two halves at a sentence boundary.

    The assignment asks for two mini-documents per page. These organization pages are
    short (about 2-3k characters) and cover a single organization, so there is no
    heading structure worth splitting on and a fixed token window would cut most pages
    at an arbitrary point. Splitting once near the midpoint, then moving to the nearest
    sentence end, keeps each half readable and roughly balanced: the first half holds
    the name and description, the second holds contact details and extra information.
    Two smaller chunks also retrieve more precisely than one large one, because a query
    matching the description is not diluted by the contact boilerplate.
    """
    midpoint = len(text) // 2
    split_at = text.find(". ", midpoint)
    if split_at == -1:  # no sentence end after the midpoint, fall back to the midpoint
        split_at = midpoint
    else:
        split_at += 1
    return text[:split_at].strip(), text[split_at:].strip()


def build_collection(collection):
    """Embed every page as two chunks, in batches."""
    ids, documents, metadatas = [], [], []
    for html_path in sorted(DATA_FOLDER.glob("*.html")):
        name, text = extract_text_from_html(html_path)
        if not text:
            continue
        for part, chunk in enumerate(split_into_two(text), start=1):
            if not chunk:
                continue
            ids.append(f"{html_path.stem}::{part}")
            # Each chunk carries the organization name so the second half is still
            # identifiable on its own once it is retrieved out of context.
            documents.append(f"{name}\n\n{chunk}")
            metadatas.append({"organization": name, "filename": html_path.name, "part": part})

    client = st.session_state.openai_client
    for start in range(0, len(ids), EMBED_BATCH):
        stop = start + EMBED_BATCH
        response = client.embeddings.create(input=documents[start:stop], model=EMBEDDING_MODEL)
        collection.add(
            ids=ids[start:stop],
            documents=documents[start:stop],
            embeddings=[item.embedding for item in response.data],
            metadatas=metadatas[start:stop],
        )


def create_hw4_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    # Only build it the first time: on later runs the collection is already populated.
    if collection.count() == 0:
        build_collection(collection)
    return collection


# Built once per session so the pages are not re-embedded on every rerun.
if "HW4_VectorDB" not in st.session_state:
    st.session_state.HW4_VectorDB = create_hw4_collection()


def query_collection(query, n_results=5):
    client = st.session_state.openai_client
    response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    return st.session_state.HW4_VectorDB.query(
        query_embeddings=[response.data[0].embedding],
        n_results=n_results,
    )


def build_context(results):
    """Label each retrieved chunk with its organization so the bot can cite it."""
    return "\n\n".join(
        f"--- {metadata['organization']} ---\n{document}"
        for metadata, document in zip(results["metadatas"][0], results["documents"][0])
    )


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

    # Retrieve the organization pages most relevant to this question and hand them to
    # the LLM alongside the recent conversation.
    context = build_context(query_collection(prompt))
    buffered_messages = st.session_state.messages[-(BUFFER_TURNS * 2 + 1):]
    messages_to_send = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Student organization context:\n{context}"},
    ] + buffered_messages

    client = st.session_state.openai_client
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages_to_send,
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
