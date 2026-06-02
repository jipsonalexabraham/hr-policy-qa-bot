# HR Policy / Document Q&A Bot

A local app that lets you upload company HR PDFs and ask questions such as "what is the leave policy?" It demonstrates the core RAG flow: PDF parsing, chunking, retrieval, prompt construction, and answer generation.

## Features

- Upload one or more HR policy PDFs.
- Extract text from each PDF page.
- Split policy text into overlapping chunks.
- Retrieve relevant chunks with TF-IDF similarity.
- Show cited source PDFs and page numbers.
- Run fully offline without an OpenAI API key.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

### Streamlit UI

```powershell
streamlit run streamlit_app.py
```

### Flask UI

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## How It Works

1. PDFs are uploaded to `uploads/`.
2. `pypdf` extracts page text.
3. Text is normalized and split into overlapping word chunks.
4. `TfidfVectorizer` embeds chunks into a searchable matrix.
5. User questions are embedded with the same vectorizer.
6. Cosine similarity retrieves the most relevant chunks.
7. The app returns the best local excerpts with source document and page citations.

## Interview Talking Points

- Chunk size and overlap affect recall and answer quality.
- Retrieval quality can be upgraded from TF-IDF to vector embeddings.
- The local answer mode avoids API keys and external network calls.
- Source citations reduce hallucination risk and make HR answers auditable.
- A production version should add authentication, persistent vector storage, document deletion, access control, and OCR for scanned PDFs.
