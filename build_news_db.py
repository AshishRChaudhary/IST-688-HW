"""Build the HW7 news vector database from Data/news.csv.

Run this once, before the app:  python build_news_db.py

HW7.py only ever loads the collection this writes, so the deployed app starts
without embedding anything.
"""

import csv
import tomllib
from pathlib import Path

import chromadb
from openai import OpenAI

REPO_DIR = Path(__file__).parent
CSV_PATH = REPO_DIR / "Data" / "news.csv"
CHROMA_PATH = str(REPO_DIR / "ChromaDB_for_HW7")
COLLECTION_NAME = "HW7Collection"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH = 100  # embed in batches; one call per article would be ~1300 requests


def read_articles():
    with CSV_PATH.open(encoding="utf-8", errors="replace", newline="") as f:
        return [row for row in csv.DictReader(f) if (row.get("Document") or "").strip()]


def main():
    secrets = tomllib.loads((REPO_DIR / ".streamlit" / "secrets.toml").read_text(encoding="utf-8"))
    client = OpenAI(api_key=secrets["OPENAI_API_KEY"])

    articles = read_articles()
    print(f"{len(articles)} articles from {CSV_PATH.name}")

    # One article per document: the Document column is a headline plus a short
    # description, so there is nothing long enough to be worth splitting.
    ids, documents, metadatas = [], [], []
    for i, article in enumerate(articles):
        ids.append(f"article-{i}")
        documents.append(
            f"{article['company_name']} ({article['Date'][:10]})\n{article['Document']}"
        )
        metadatas.append(
            {
                "company_name": article["company_name"],
                "date": article["Date"][:10],
                "url": article["URL"],
            }
        )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() > 0:
        print(f"{COLLECTION_NAME} already has {collection.count()} articles; nothing to do.")
        return

    for start in range(0, len(ids), EMBED_BATCH):
        stop = start + EMBED_BATCH
        response = client.embeddings.create(input=documents[start:stop], model=EMBEDDING_MODEL)
        collection.add(
            ids=ids[start:stop],
            documents=documents[start:stop],
            embeddings=[item.embedding for item in response.data],
            metadatas=metadatas[start:stop],
        )
        print(f"  embedded {min(stop, len(ids))}/{len(ids)}")

    print(f"done: {collection.count()} articles in {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
