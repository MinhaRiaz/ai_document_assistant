# AI Document Assistant

A simple Streamlit RAG-style document assistant that lets users upload PDF, DOCX, TXT, and Markdown files, or load supported files from Google Drive.

It extracts the text, creates overlapping chunks, generates Sentence Transformers embeddings, stores them in a FAISS index, performs hybrid semantic + keyword search, and sends the retrieved context to Groq.

## Features

- PDF upload and extraction with page numbers
- DOCX extraction
- TXT extraction
- Markdown extraction
- Filename metadata for every document
- Page metadata where available
- Overlapping text chunking
- Sentence Transformers embeddings
- FAISS vector search
- Simple keyword search
- Hybrid semantic + keyword ranking
- Groq-powered answers
- Answers restricted to retrieved document context
- "Information not available" response when the context does not contain the answer
- Retrieved source chunks shown after every answer
- Google Drive file/folder loading
- Local upload continues to work
- Embeddings are created once per document set, not for every question
- Streamlit session state and caching reduce repeated processing
- Groq API key is stored in Streamlit Secrets

## Project files

Only three files are needed:

```text
app.py
requirements.txt
readme.md
```

## How the pipeline works

```text
Documents
   ↓
Text extraction
   ↓
Overlapping chunks
   ↓
Sentence Transformers embeddings
   ↓
FAISS vector index
   ↓
Question
   ↓
Semantic search + keyword search
   ↓
Hybrid ranking
   ↓
Top document chunks
   ↓
Groq
   ↓
Grounded answer + retrieved sources
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Add the Groq API key

Do NOT put the API key inside `app.py`.

For Streamlit Cloud, open your app settings and add this secret:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

For local development, create:

```text
.streamlit/secrets.toml
```

and put:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Never commit `.streamlit/secrets.toml` to GitHub.

## 3. Run the app

```bash
streamlit run app.py
```

## Google Drive

Paste a Google Drive file or folder link into the sidebar and click **Load from Google Drive**.

The Drive source uses `gdown`.

Supported file types:

- `.pdf`
- `.docx`
- `.txt`
- `.md`

For the simplest setup, the Drive file/folder should be accessible through its shared link.

Google Drive folders may contain unsupported files. Those files are ignored.

## How embeddings are reused

The app uses:

- `st.cache_resource` for the Sentence Transformers model
- `st.cache_data` for document embeddings
- `st.session_state` for chunks, embeddings, and the FAISS index
- A document signature to detect whether the current document set changed

Therefore, asking multiple questions about the same documents does not recreate all document embeddings.

Only the new question is embedded for each search.

## Important implementation details

### PDF

PDFs are extracted page by page, so the source can show:

```text
Filename: report.pdf
Page: 4
```

### DOCX, TXT and MD

These formats do not reliably provide page information through the simple extraction used here, so their page value is shown as:

```text
Not available
```

### Hybrid search

Semantic search uses Sentence Transformers + FAISS.

Keyword search counts occurrences of important question words in each chunk.

The final ranking combines both:

```text
Hybrid score =
    75% semantic score
    +
    25% keyword score
```

### Grounded generation

The Groq model receives:

1. The user's question
2. The retrieved document chunks

The system instruction tells the model to answer only from that context and explicitly say when the information is unavailable.

## Notes

- This is intentionally kept simple and beginner-friendly.
- FAISS uses normalized embeddings with inner-product search, which behaves like cosine similarity.
- The default chunk size is 800 characters with 150 characters of overlap.
- The default Groq model in `app.py` is `llama-3.3-70b-versatile`. If that model is unavailable for your Groq account, replace it with a currently available Groq model.
- For very large document collections, a persistent vector database would be more appropriate than keeping the index in Streamlit session state.
- Google Drive folder downloads depend on the files being accessible through the provided Drive link.

## Security

Never hard-code:

```python
GROQ_API_KEY = "..."
```

The app correctly reads the key from:

```python
st.secrets["GROQ_API_KEY"]
```

Keep your API key private and do not commit it to GitHub.
