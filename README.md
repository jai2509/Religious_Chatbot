✝️ Orthodox Comparative Theology Chatbot

An AI-powered web application that provides side-by-side theological comparisons between a user’s selected belief system and Eastern Orthodox Christianity.

The chatbot uses Retrieval-Augmented Generation (RAG) to reference real theological sources (e.g., Church Fathers, Councils, Catechisms) and generates structured, reasoned explanations using Groq’s LLaMA 3 model.# Religious_Chatbot
🧭 Project Overview

This app enables users to:

🕊️ Select or describe their current belief system (e.g., Catholic, Protestant, Muslim, Atheist).

📖 Ask about a topic (Trinity, salvation, icons, Scripture, etc.).

⚖️ Receive a step-by-step, side-by-side explanation comparing their belief system with the Orthodox perspective.

📚 Each comparison is backed by theological and historical reasoning, citing indexed source texts.

The app uses:

Groq API for fast LLM inference.

SentenceTransformers + ChromaDB for vectorized retrieval (RAG).

Streamlit for a clean, deployable web UI.
🧩 Tech Stack
| Component          | Technology                                    |
| ------------------ | --------------------------------------------- |
| Frontend           | Streamlit                                     |
| Backend            | Python (FastAPI-style logic inside Streamlit) |
| LLM                | [Groq API](https://groq.com/) — LLaMA 3-70B   |
| Embeddings         | SentenceTransformers (`all-MiniLM-L6-v2`)     |
| Vector DB          | Chroma (PersistentClient with `pysqlite3`)    |
| File Parsing       | PyMuPDF (`fitz`)                              |
| Environment Config | `python-dotenv`                               |
🚀 Features

✅ Belief selection UI — Catholic, Protestant, Muslim, Atheist, etc.
✅ Topic input — user can ask about any theological question.
✅ RAG pipeline — retrieves relevant content from uploaded or stored PDFs.
✅ LLM reasoning — generates side-by-side Markdown table with citations and summary.
✅ File upload & auto-indexing — add new theological PDFs dynamically.
✅ Persistent vector DB — data saved locally in chroma_db/ for reuse.
✅ Streamlit-ready deployment — one-click deploy to Streamlit Cloud
.🧰 Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/orthodox-chatbot.git
cd orthodox-chatbot
2️⃣ Install dependencies

Make sure you have Python 3.9+ installed, then:
pip install -r requirements.txt
3️⃣ Add environment variables

Create a file named .env in the project root (do not commit it):
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=4
CHROMA_DB_DIR=chroma_db
🕮 Adding Theological Sources

The chatbot uses RAG — it relies on your PDF source texts (e.g., catechisms, patristic writings, ecumenical council documents).

You can:

Place PDFs in the repo root or inside a folder named pdfs/, or

Upload them dynamically from the app sidebar (Upload theological PDFs).

They will be automatically indexed into the Chroma vector database.
streamlit run app.py
Then open: http://localhost:8501

🌐 Deploy to Streamlit Cloud

Push this repo to GitHub (excluding your .env).

Go to Streamlit Cloud
 → “Deploy an app”.

Connect your GitHub repo.

In the app Settings → Secrets, add:
GROQ_API_KEY = "your_groq_api_key_here"
Deploy!

The app will automatically build, index PDFs, and go live.

🧠 How It Works

Document Ingestion

Extracts text from PDFs using PyMuPDF.

Splits into chunks (CHUNK_SIZE, CHUNK_OVERLAP).

Embeds with SentenceTransformers and stores vectors in Chroma.

Retrieval

When a user asks a question, the query is embedded.

Top-K relevant passages are retrieved from the database.

LLM Reasoning (Groq)

A custom system prompt asks LLaMA 3 to compare the user’s belief with Orthodoxy.

Generates a two-column Markdown table + summary + disclaimer.
🧾 Example Output

Input

Belief system: Protestant
Topic: The role of icons in worship
| User’s Belief                                    | Eastern Orthodox Perspective                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Icons are often avoided due to fear of idolatry. | Icons are venerated, not worshiped; they represent the Incarnation (John 1:14). [1][2]            |
| Scripture alone is the guide for worship forms.  | Tradition and Scripture are co-witnesses to faith; the 7th Ecumenical Council defended icons. [3] |
Summary:
While Protestants often see icons as potential distractions from worship, Orthodoxy views them as manifestations of divine reality through matter.
Note: Educational only; not pastoral advice.

🛠 Maintenance & Customization

Update theological sources by dropping new PDFs into /pdfs and using Rebuild Index in the sidebar.

To change the LLM prompt, edit the compose_prompt() function in app.py.

Adjust RAG chunk sizes or retrieval count (CHUNK_SIZE, TOP_K) in .env.

⚖️ License

MIT License © 2025 — Developed by an AI Research & Data Science Team.
