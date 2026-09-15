import io
import os
import re
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Document Assistant")
st.caption("Upload documents or load supported files from Google Drive, then ask questions using hybrid search.")


# -----------------------------
# Session state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "index" not in st.session_state:
    st.session_state.index = None

if "sources_signature" not in st.session_state:
    st.session_state.sources_signature = None

if "documents" not in st.session_state:
    st.session_state.documents = []


# -----------------------------
# Cached models / embeddings
# -----------------------------
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def create_embeddings(texts):
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


# -----------------------------
# Extraction functions
# -----------------------------
def extract_pdf(file_bytes, filename):
    """Return one record per PDF page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            records.append(
                {
                    "text": text.strip(),
                    "filename": filename,
                    "page": page_number,
                }
            )

    return records


def extract_docx(file_bytes, filename):
    """Extract DOCX paragraphs. DOCX does not reliably expose page numbers."""
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [
        {
            "text": text.strip(),
            "filename": filename,
            "page": None,
        }
    ]


def extract_txt(file_bytes, filename):
    """Extract plain text."""
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        {
            "text": text.strip(),
            "filename": filename,
            "page": None,
        }
    ]


def extract_md(file_bytes, filename):
    """Extract Markdown as text."""
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        {
            "text": text.strip(),
            "filename": filename,
            "page": None,
        }
    ]


def extract_document(file_bytes, filename):
    """Choose the correct extractor from the file extension."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# -----------------------------
# Chunking
# -----------------------------
def chunk_records(records, chunk_size=800, overlap=150):
    """
    Split text into overlapping character chunks.
    Filename and page metadata stay attached to every chunk.
    """
    chunks = []

    for record in records:
        text = record["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "filename": record["filename"],
                        "page": record["page"],
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# -----------------------------
# Hybrid search
# -----------------------------
def important_words(question):
    stop_words = {
        "what", "when", "where", "which", "who", "why", "how",
        "is", "are", "was", "were", "the", "a", "an", "and",
        "or", "of", "to", "in", "on", "for", "with", "from",
        "about", "this", "that", "can", "could", "should",
        "do", "does", "did", "i", "you", "it", "my", "your",
    }

    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    return [word for word in words if word not in stop_words and len(word) > 2]


def keyword_scores(question, chunks):
    words = important_words(question)
    scores = []

    for chunk in chunks:
        text = chunk["text"].lower()
        if not words:
            scores.append(0.0)
            continue

        matches = sum(text.count(word) for word in words)
        scores.append(matches / len(words))

    return np.array(scores, dtype="float32")


def hybrid_search(question, top_k=5):
    if not st.session_state.chunks or st.session_state.index is None:
        return []

    model = get_embedding_model()

    # Semantic score from FAISS cosine similarity.
    question_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    semantic_scores, semantic_ids = st.session_state.index.search(
        question_vector,
        min(top_k * 3, len(st.session_state.chunks)),
    )

    semantic_scores = semantic_scores[0]
    semantic_ids = semantic_ids[0]

    # Keyword scores for every chunk.
    kw_scores = keyword_scores(question, st.session_state.chunks)

    # Normalize semantic scores to 0..1.
    if len(semantic_scores) > 0:
        sem_min = float(np.min(semantic_scores))
        sem_max = float(np.max(semantic_scores))
        if sem_max > sem_min:
            normalized_semantic = (semantic_scores - sem_min) / (sem_max - sem_min)
        else:
            normalized_semantic = np.ones_like(semantic_scores)
    else:
        normalized_semantic = np.array([])

    # Combine semantic + keyword scores.
    candidates = []

    for rank, chunk_id in enumerate(semantic_ids):
        if chunk_id < 0:
            continue

        combined = (
            0.75 * float(normalized_semantic[rank])
            + 0.25 * float(kw_scores[chunk_id])
        )

        candidates.append(
            {
                "chunk_id": int(chunk_id),
                "score": combined,
                **st.session_state.chunks[chunk_id],
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


# -----------------------------
# Document processing
# -----------------------------
def build_pipeline(documents):
    """Extract, chunk, embed, and build the FAISS index once."""
    all_records = []

    for filename, file_bytes in documents:
        records = extract_document(file_bytes, filename)
        all_records.extend(records)

    chunks = chunk_records(all_records)

    if not chunks:
        raise ValueError("No readable text was found in the supplied documents.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.index = index

    return all_records, chunks


def documents_signature(documents):
    """Create a stable signature so unchanged documents are not reprocessed."""
    import hashlib

    hasher = hashlib.sha256()

    for filename, file_bytes in documents:
        hasher.update(filename.encode("utf-8"))
        hasher.update(file_bytes)

    return hasher.hexdigest()


# -----------------------------
# Google Drive
# -----------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_from_google_drive(url):
    """
    Load a public/shared Google Drive file or folder.
    gdown handles Google Drive download links.
    """
    temp_dir = tempfile.mkdtemp(prefix="drive_docs_")
    documents = []

    if "/folders/" in url:
        downloaded_dir = gdown.download_folder(
            url,
            output=temp_dir,
            quiet=True,
            use_cookies=False,
        )

        if downloaded_dir:
            root = Path(downloaded_dir)
            files = [p for p in root.rglob("*") if p.is_file()]
        else:
            files = []
    else:
        output_path = Path(temp_dir) / "drive_file"
        downloaded = gdown.download(
            url,
            output=str(output_path),
            quiet=True,
        )

        files = [Path(downloaded)] if downloaded else []

    for path in files:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            documents.append((path.name, path.read_bytes()))
        except OSError:
            continue

    return documents


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=5,
    )

    st.divider()

    st.subheader("Google Drive")
    drive_url = st.text_input(
        "Paste a public/shared Drive file or folder link",
        placeholder="https://drive.google.com/...",
    )

    load_drive = st.button(
        "Load from Google Drive",
        use_container_width=True,
    )

    st.caption(
        "Drive links must be accessible to the account/session used for downloading. "
        "Supported: PDF, DOCX, TXT, MD."
    )


# -----------------------------
# Local uploads
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

local_documents = []
for uploaded_file in uploaded_files:
    local_documents.append((uploaded_file.name, uploaded_file.getvalue()))


# -----------------------------
# Drive loading
# -----------------------------
if load_drive:
    if not drive_url.strip():
        st.warning("Please paste a Google Drive file or folder link first.")
    else:
        with st.spinner("Loading files from Google Drive..."):
            try:
                drive_documents = load_from_google_drive(drive_url.strip())

                if not drive_documents:
                    st.warning(
                        "No supported files were found. Make sure the Drive link is "
                        "public/shared and contains PDF, DOCX, TXT, or MD files."
                    )
                else:
                    # Store Drive documents separately so they remain available
                    # after the button reruns the Streamlit script.
                    st.session_state.drive_documents = drive_documents
                    st.success(f"Loaded {len(drive_documents)} supported file(s) from Drive.")
            except Exception as error:
                st.error(f"Could not load the Drive link: {error}")


if "drive_documents" not in st.session_state:
    st.session_state.drive_documents = []


# Combine local + Drive sources.
all_documents = local_documents + st.session_state.drive_documents


# -----------------------------
# Process only when sources change
# -----------------------------
if all_documents:
    current_signature = documents_signature(all_documents)

    if current_signature != st.session_state.sources_signature:
        with st.spinner("Extracting, chunking, embedding, and indexing documents..."):
            try:
                records, chunks = build_pipeline(all_documents)
                st.session_state.sources_signature = current_signature

                st.success(
                    f"Processed {len(all_documents)} document(s) and created "
                    f"{len(chunks)} chunk(s)."
                )
            except Exception as error:
                st.error(f"Document processing failed: {error}")

# -----------------------------
# Document information
# -----------------------------
if st.session_state.chunks:
    st.subheader("Document Information")

    document_names = sorted(
        {chunk["filename"] for chunk in st.session_state.chunks}
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(document_names))
    col2.metric("Chunks", len(st.session_state.chunks))
    col3.metric(
        "Embedding size",
        st.session_state.embeddings.shape[1]
        if st.session_state.embeddings is not None
        else 0,
    )

    with st.expander("Show extracted/chunked document information"):
        for name in document_names:
            file_chunks = [
                chunk for chunk in st.session_state.chunks
                if chunk["filename"] == name
            ]

            pages = sorted(
                {
                    chunk["page"]
                    for chunk in file_chunks
                    if chunk["page"] is not None
                }
            )

            if pages:
                page_info = f"PDF pages represented: {pages[0]}–{pages[-1]}"
            else:
                page_info = "Page number: not available"

            st.write(
                f"**{name}** — {len(file_chunks)} chunks — {page_info}"
            )


# -----------------------------
# Question + Groq
# -----------------------------
st.subheader("Ask your documents")

question = st.text_input(
    "Question",
    placeholder="Ask something about the uploaded documents...",
)

ask = st.button("Ask AI", type="primary")

if ask:
    if not st.session_state.chunks:
        st.warning("Please upload a document or load one from Google Drive first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    elif "GROQ_API_KEY" not in st.secrets:
        st.error(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets before asking questions."
        )
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                results = hybrid_search(question, top_k=top_k)

                if not results:
                    st.info("No relevant document chunks were found.")
                else:
                    context_parts = []

                    for number, result in enumerate(results, start=1):
                        page = (
                            f"page {result['page']}"
                            if result["page"] is not None
                            else "page not available"
                        )

                        context_parts.append(
                            f"[Source {number}]\n"
                            f"Filename: {result['filename']}\n"
                            f"Page: {page}\n"
                            f"Text:\n{result['text']}"
                        )

                    context = "\n\n".join(context_parts)

                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an AI Document Assistant. "
                                    "Answer the user's question ONLY from the supplied "
                                    "document context. Do not use outside knowledge. "
                                    "If the answer is not present in the context, say: "
                                    "'The information is not available in the provided documents.' "
                                    "Be clear and concise."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Question:\n{question}\n\n"
                                    f"Document context:\n{context}"
                                ),
                            },
                        ],
                        temperature=0,
                    )

                    answer = response.choices[0].message.content

                    st.markdown("### Answer")
                    st.write(answer)

                    st.markdown("### Retrieved Sources")

                    for number, result in enumerate(results, start=1):
                        page = (
                            str(result["page"])
                            if result["page"] is not None
                            else "Not available"
                        )

                        with st.expander(
                            f"{number}. {result['filename']} — Page: {page}"
                        ):
                            st.caption(
                                f"Hybrid relevance score: {result['score']:.3f}"
                            )
                            st.write(result["text"])

            except Exception as error:
                st.error(f"Could not generate the answer: {error}")


st.divider()
st.caption(
    "Pipeline: Extract → Chunk → Embed → FAISS + Keyword Hybrid Search → Groq"
)
