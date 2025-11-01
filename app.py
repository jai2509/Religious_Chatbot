# app.py
# Orthodox Comparative Theology Chatbot (Streamlit)
# - RAG with sentence-transformers + chromadb PersistentClient
# - Groq LLM as backend
# - Indexes PDFs in repo root and pdfs/; supports uploads via UI
# - Produces side-by-side Markdown table comparing user's belief and Eastern Orthodox perspective

# SQLite patch for environments with old sqlite (must be first)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import time

# Load env
load_dotenv()

# ------------- Configuration -------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 4))
DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")

st.set_page_config(page_title="Orthodox Comparative Chatbot", page_icon="✝️", layout="wide")

st.title("Orthodox Comparative Chatbot")
st.markdown(
    "**Educational only — not pastoral advice.**\n\n"
    "Select your belief system, ask a topic (e.g., Trinity, icons, salvation), and get a side-by-side comparison "
    "between your group's typical belief and the Eastern Orthodox perspective (with reasoning and source citations)."
)

# ------------- Embeddings (cached) -------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

embedder = get_embedder()

# ------------- Chroma PersistentClient & collection -------------
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    name="theology_docs",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL)
)

# ------------- PDF -> text utilities -------------
def pdf_to_text(path: Path) -> str:
    text_pages = []
    try:
        with fitz.open(path) as doc:
            for p in doc:
                txt = p.get_text()
                if txt:
                    text_pages.append(txt)
    except Exception as e:
        st.warning(f"Could not read PDF {path.name}: {e}")
    return "\n".join(text_pages)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# ------------- Indexing PDFs -------------
def index_pdfs(search_path: Path):
    try:
        existing_ids = set(collection.get()["ids"])
    except Exception:
        existing_ids = set()
    for pdf in search_path.glob("*.pdf"):
        txt = pdf_to_text(pdf)
        if not txt.strip():
            continue
        chunks = chunk_text(txt)
        docs, metas, ids = [], [], []
        for i, c in enumerate(chunks):
            doc_id = f"{pdf.name}_{i}"
            if doc_id in existing_ids:
                continue
            docs.append(c)
            metas.append({"source": pdf.name})
            ids.append(doc_id)
        if docs:
            collection.add(documents=docs, metadatas=metas, ids=ids)

def discover_and_index():
    root = Path(".")
    root_pdfs = list(root.glob("*.pdf"))
    pdfs_folder = Path("pdfs")
    folder_pdfs = list(pdfs_folder.glob("*.pdf")) if pdfs_folder.exists() else []
    all_pdfs = root_pdfs + folder_pdfs
    if not all_pdfs:
        return False, []
    index_pdfs(root)
    if pdfs_folder.exists():
        index_pdfs(pdfs_folder)
    return True, [p.name for p in all_pdfs]

# initial indexing
indexed_any, indexed_files = discover_and_index()
if indexed_any:
    st.sidebar.success(f"Indexed PDFs: {len(indexed_files)}")
else:
    st.sidebar.info("No PDFs indexed yet. Place PDFs in repo root or 'pdfs/' or upload below.")

# ------------- Upload UI -------------
st.sidebar.header("Upload theological PDFs")
uploaded = st.sidebar.file_uploader("Upload one or more PDFs to add sources", accept_multiple_files=True, type=["pdf"])
if uploaded:
    save_folder = Path("pdfs")
    save_folder.mkdir(exist_ok=True)
    for f in uploaded:
        dest = save_folder / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success("Saved uploaded PDFs to /pdfs; re-indexing...")
    time.sleep(0.3)
    index_pdfs(save_folder)
    st.experimental_rerun()

if st.sidebar.button("Rebuild index (delete & re-index)"):
    try:
        client.delete_collection(name="theology_docs")
    except Exception:
        pass
    # recreate and reindex
    collection = client.get_or_create_collection(
        name="theology_docs",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL)
    )
    discover_and_index()
    st.sidebar.success("Rebuilt index.")
    st.experimental_rerun()

# ------------- Retrieval -------------
def retrieve(query: str, k=TOP_K):
    try:
        results = collection.query(query_texts=[query], n_results=k)
        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({"text": doc, "source": meta.get("source", "unknown")})
        return docs
    except Exception:
        return []

# ------------- Prompt composition -------------
def compose_prompt(topic: str, user_belief: str, contexts: list):
    # create numbered context list for model citations
    ctx_lines = []
    for i, c in enumerate(contexts):
        excerpt = c.get("text", "")[:1000].replace("\n", " ")
        src = c.get("source", "unknown")
        ctx_lines.append(f"[{i+1}] Source: {src}\n{excerpt}")
    ctx_text = "\n\n".join(ctx_lines) if ctx_lines else "No context documents available."

    prompt = (
        "You are a careful theological assistant focused on Eastern Orthodox Christianity.\n"
        "Use ONLY the provided context items as evidence. When you reference evidence, label it like [1], [2], ... matching the context list.\n\n"
        f"CONTEXT:\n{ctx_text}\n\n"
        f"User belief group: {user_belief}\n"
        f"Question / Topic: {topic}\n\n"
        "Produce a concise side-by-side comparison in Markdown table format with TWO columns:\n"
        "- Left column header: \"User's Belief\"\n"
        "- Right column header: \"Eastern Orthodox Perspective\"\n\n"
        "Each cell should contain bullet points (2-6 short bullets) explaining the key positions. "
        "In the Orthodox column, include theological and historical reasoning and cite context items like [1], [2].\n\n"
        "After the table, provide a 2-3 sentence summary and finish with this exact disclaimer: \"Note: Educational only; not pastoral advice.\""
    )
    return prompt

# ------------- Groq call -------------
def call_groq_chat(prompt: str):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": "You are a precise theological assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

# ------------- Main UI -------------
st.sidebar.header("Query settings")
user_group = st.sidebar.selectbox(
    "Select your belief group",
    ["Catholic", "Protestant", "Evangelical", "Anglican", "Pentecostal", "Orthodox (other)", "Muslim", "Jewish", "Atheist/Agnostic", "Other / Describe"]
)
if user_group == "Other / Describe":
    user_group = st.sidebar.text_input("Describe your belief or group", value="")

topic = st.text_input("Enter the theological topic/question (e.g., Trinity, icons, salvation):")
if st.button("Compare beliefs"):
    if not topic.strip():
        st.error("Please enter a topic or question.")
    else:
        with st.spinner("Retrieving context..."):
            contexts = retrieve(topic, k=TOP_K)
        st.subheader("Retrieved passages (numbered)")
        if contexts:
            for i, c in enumerate(contexts, start=1):
                st.markdown(f"**[{i}] {c.get('source','unknown')}**")
                st.write(c.get("text","")[:600] + ("..." if len(c.get("text",""))>600 else ""))
        else:
            st.info("No passages found; the model will answer without direct citations.")

        prompt = compose_prompt(topic, user_group, contexts)
        with st.spinner("Generating comparison (Groq)..."):
            try:
                result = call_groq_chat(prompt)
            except Exception as e:
                st.error(f"Groq API error: {e}")
                result = None

        if result:
            st.subheader("Side-by-side Comparison")
            st.markdown(result)
        else:
            st.error("No result returned from the model.")
