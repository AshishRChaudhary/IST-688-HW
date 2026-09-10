import anthropic
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

st.title("💬 HW3 - Chatbot that discusses a URL")
st.write(
    "This chatbot answers questions about one or two web pages. Paste up to two URLs in "
    "the sidebar and pick which LLM to use. The pages are read and placed in a system "
    "prompt that is sent with every question, so the chatbot never loses the documents. "
    "For conversation memory it keeps the last 6 messages (3 question-and-answer "
    "exchanges) as they are. Once 6 messages build up, they are replaced by a short "
    "summary written by the same LLM, and that summary travels with each later question. "
    "This keeps the conversation going without resending the whole history every time."
)

MODELS = {
    "OpenAI - gpt-5.6-luna": ("openai", "gpt-5.6-luna"),
    "Anthropic - claude-fable-5-1": ("anthropic", "claude-fable-5-1"),
}

SUMMARY_AFTER = 6  # messages, i.e. 3 user-assistant exchanges

RULES = (
    "You are a helpful assistant who answers questions about the documents below. "
    "Base your answers on the documents. If the documents do not cover something, say "
    "so instead of guessing."
)


def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_documents(urls):
    """Read each URL once and keep it, so every chat turn does not refetch."""
    documents = {}
    for url in urls:
        if url:
            documents[url] = read_url_content(url)
    return documents


def build_system_prompt(documents, summary):
    parts = [RULES]
    for i, (url, text) in enumerate(documents.items(), start=1):
        parts.append(f"Document {i} ({url}):\n{text}")
    if summary:
        parts.append(f"Summary of the earlier conversation:\n{summary}")
    return "\n\n---\n\n".join(parts)


def stream_answer(vendor, model, system_prompt, conversation):
    if vendor == "openai":
        messages = [{"role": "system", "content": system_prompt}] + conversation
        stream = st.session_state.openai.chat.completions.create(
            model=model, messages=messages, stream=True
        )
        return st.write_stream(stream)

    with st.session_state.anthropic.messages.stream(
        model=model, max_tokens=2000, system=system_prompt, messages=conversation
    ) as stream:
        return st.write_stream(stream.text_stream)


def summarize(vendor, model, previous_summary, conversation):
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
    instruction = (
        "Summarize this conversation in under 150 words, keeping what matters for "
        "answering follow-up questions."
    )
    if previous_summary:
        instruction += f"\n\nSummary so far:\n{previous_summary}"
    prompt = f"{instruction}\n\nConversation:\n{transcript}"

    if vendor == "openai":
        response = st.session_state.openai.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    response = st.session_state.anthropic.messages.create(
        model=model, max_tokens=1000, messages=[{"role": "user", "content": prompt}]
    )
    return "".join(b.text for b in response.content if b.type == "text")


if "openai" not in st.session_state:
    st.session_state.openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
if "anthropic" not in st.session_state:
    st.session_state.anthropic = anthropic.Anthropic(
        api_key=st.secrets["ANTHROPIC_API_KEY"]
    )

# messages is the whole conversation, shown to the user. recent is what is actually
# sent to the LLM: it is emptied into summary once it reaches SUMMARY_AFTER messages.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about the URLs in the sidebar."}
    ]
if "recent" not in st.session_state:
    st.session_state.recent = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

with st.sidebar:
    st.header("URLs")
    url1 = st.text_input("First URL", placeholder="https://example.com")
    url2 = st.text_input("Second URL (optional)", placeholder="https://example.com")

    st.header("LLM")
    model_label = st.selectbox("Model", list(MODELS.keys()))
    vendor, model = MODELS[model_label]

    if st.session_state.summary:
        st.caption("Earlier conversation is stored as a summary.")
    st.caption(f"{len(st.session_state.recent)} of {SUMMARY_AFTER} messages buffered")

documents = load_documents((url1, url2))
unreadable = [url for url, text in documents.items() if not text]
for url in unreadable:
    st.error(f"Could not read {url}. Check the address and try again.")
documents = {url: text for url, text in documents.items() if text}

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if not documents:
    st.info("Add a URL in the sidebar to start chatting about it.")
elif prompt := st.chat_input("Ask a question about the documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.recent.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # The system prompt is rebuilt every turn, so the documents are never discarded.
    system_prompt = build_system_prompt(documents, st.session_state.summary)

    with st.chat_message("assistant"):
        answer = stream_answer(vendor, model, system_prompt, st.session_state.recent)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.recent.append({"role": "assistant", "content": answer})

    # Conversation memory: after 3 exchanges, fold the buffer into a summary.
    if len(st.session_state.recent) >= SUMMARY_AFTER:
        st.session_state.summary = summarize(
            vendor, model, st.session_state.summary, st.session_state.recent
        )
        st.session_state.recent = []
        st.rerun()
