# Multimodal RAG on World Bank Trust Fund Reports

This project is a **retrieval‑augmented generation (RAG)** proof of concept that lets analysts explore large PDF reports through a conversational Streamlit UI. It pairs **Cohere’s vision embeddings** with a **local FAISS index** so we can retrieve the most relevant report pages, then hands those page images to **OpenAI Vision** to compose grounded answers that cite the exact chart/table and source document.

> ⚠️ This is an internal enablement project—there is no production hardening, auth, or multi-user coordination.

---

## What you get

- **Document ingestion pipeline** – Drop PDFs into `data/raw/source_docs/`, run one command, and every page becomes a PNG plus a normalized embedding.
- **Local semantic search** – FAISS keeps all embeddings on disk at `data/processed/vector_store/`, so queries are fast and offline apart from the Cohere/OpenAI calls.
- **Grounded visual answers** – The app retrieves *K* relevant pages (user controlled via slider), sends them to GPT‑4.1‑mini with strict instructions, and the answer cites the page filename + report name.
- **Session persistence** – Every chat is saved under `data/chat/sessions/` so conversations survive Streamlit reloads.
- **Polished Streamlit experience** – Background imagery, a PDF uploader, “start new chat” control, and an image gallery that shows the exact figures used for an answer.

---

## Repository layout

```text
.
├── src/
│   ├── app.py                    # Streamlit UI + workflow orchestration
│   ├── config.py                 # Centralized paths and API client setup
│   ├── pdf_processing_embedding.py
│   ├── vision_query.py           # FAISS lookup + multimodal prompt builder
│   ├── faiss_utils.py            # Index load/save operations
│   ├── utils.py                  # Hashing, PDF → image conversion, helpers
│   └── chat_history.py           # Session ID + history persistence helpers
├── data/
│   ├── raw/source_docs/          # Input PDFs (World Bank Annual Reports, etc.)
│   ├── processed/images/         # Generated page PNGs
│   ├── processed/vector_store/   # FAISS index + filename map
│   ├── cache/hashes/             # SHA256 per PDF so we skip unchanged files
│   └── chat/sessions/            # JSON transcripts per Streamlit session
├── assets/                       # Background + avatar images for UI
├── notebooks/                    # Exploratory / throwaway experiments
└── requirements.txt
```

All executable code lives in `src/`, everything generated at runtime lives under `data/`, and UI assets are isolated so the git history stays clean.

---

## Prerequisites

1. **Python 3.9+**
2. **Cohere and OpenAI API keys**
3. **Pip packages**
   ```bash
   pip install -r requirements.txt
   ```
   (PyMuPDF handles PDF rendering—no external Poppler binary is required.)

4. **Environment variables** – create `.env` in the repo root:
   ```ini
   COHERE_API_KEY=your_cohere_key
   OPENAI_API_KEY=your_openai_key
   ```

---

## Ingesting / refreshing documents

1. Place new PDFs in `data/raw/source_docs/`.
2. Run the ingestion script (needs Cohere access because it generates embeddings):
   ```bash
   PYTHONPATH=src python3 - <<'PY'
   from pdf_processing_embedding import process_pdfs_and_embed_pages
   from config import co
   process_pdfs_and_embed_pages(co)
   PY
   ```
   - Each PDF page is rendered to `data/processed/images/<pdf>_page<N>.png`.
   - Embeddings are appended to the FAISS index at `data/processed/vector_store/`.
   - `data/cache/hashes/pdf_hashes.json` tracks SHA256 hashes so reruns skip unchanged PDFs automatically.

Re-run the command any time you add or update source documents.

---

## Running the Streamlit app

```bash
streamlit run src/app.py
```

Key UI elements:

- **PDF uploader (sidebar):** Quickly add a single report without leaving the app. Uploaded files run through the same ingestion pipeline and update the FAISS index on the fly.
- **Question input + retrieval slider:** Ask natural language questions and choose how many reference pages (`top_k` slider) should feed the answer.
- **Chat history pane:** Displays user/bot messages with custom avatars. Conversations persist between runs.
- **Relevant images gallery:** Shows every retrieved page; clicking a thumbnail opens a modal with a zoomed figure so you can visually confirm the answer.

Under the hood the button handler:
1. Looks up `top_k` nearest embeddings via FAISS.
2. Sends those page images plus a structured instruction prompt to OpenAI.
3. Stores the question, grounded answer, and image paths to the active chat session.

---

## Prompting approach

`vision_query.answer_question_about_images` composes a multimodal prompt that:
- Sets the system role to “analyze scanned pages from World Bank Trust Fund annual reports.”
- Provides a reference list for each retrieved page (`filename`, inferred document name, and page number).
- Instructs the model to cite all supporting pages, quote numbers exactly, and respond “No evidence in provided images.” when nothing relevant is found.

This keeps answers concise, verifiable, and tightly grounded to the retrieved context.

---

## Known limitations

- Local single-user workflow; there is no authentication or concurrency control.
- FAISS index lives on disk—shipping this anywhere else requires copying the whole `data/processed/` directory.
- No guardrails for PII / secrets beyond whatever Cohere and OpenAI provide.
- Error logs print in the Streamlit console; there is no centralized monitoring.

Despite those constraints, the app is ideal for prototyping multimodal RAG flows, validating document coverage, and demonstrating how grounded answers can cite exact charts/tables within lengthy PDF reports.

---


