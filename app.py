import re
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
TOP_K = 1

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@dataclass
class Chunk:
    text: str
    filename: str
    page: int
    chunk_id: int


class HRKnowledgeBase:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.idf: Dict[str, float] = {}
        self.vectors: List[Dict[str, float]] = []

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and bool(self.vectors)

    def clear(self) -> None:
        self.chunks = []
        self.idf = {}
        self.vectors = []

    def add_pdf(self, pdf_path: Path) -> int:
        reader = PdfReader(str(pdf_path))
        added = 0

        for page_index, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if not text:
                continue

            for chunk_id, chunk_text in enumerate(chunk_text_by_words(text), start=1):
                self.chunks.append(
                    Chunk(
                        text=chunk_text,
                        filename=pdf_path.name,
                        page=page_index,
                        chunk_id=chunk_id,
                    )
                )
                added += 1

        self._rebuild_index()
        return added

    def search(self, question: str, top_k: int = TOP_K) -> List[dict]:
        if not self.ready:
            return []

        query_vector = vectorize(question, self.idf)
        scores = [cosine_similarity(query_vector, vector) for vector in self.vectors]
        best_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]

        results = []
        for index in best_indices:
            score = scores[index]
            if score <= 0:
                continue
            chunk = self.chunks[index]
            results.append(
                {
                    "text": chunk.text,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "score": round(score, 4),
                }
            )

        return results

    def _rebuild_index(self) -> None:
        if not self.chunks:
            self.idf = {}
            self.vectors = []
            return

        tokenized_chunks = [tokenize(chunk.text) for chunk in self.chunks]
        document_frequency = Counter()
        for tokens in tokenized_chunks:
            document_frequency.update(set(tokens))

        total_documents = len(tokenized_chunks)
        self.idf = {
            term: math.log((1 + total_documents) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = [vectorize_tokens(tokens, self.idf) for tokens in tokenized_chunks]


knowledge_base = HRKnowledgeBase()


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower())
    bigrams = [f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1)]
    return words + bigrams


def vectorize(text: str, idf: Dict[str, float]) -> Dict[str, float]:
    return vectorize_tokens(tokenize(text), idf)


def vectorize_tokens(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(token for token in tokens if token in idf)
    if not counts:
        return {}

    max_count = max(counts.values())
    return {
        term: (count / max_count) * idf[term]
        for term, count in counts.items()
    }


def cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    shared_terms = set(left) & set(right)
    dot_product = sum(left[term] * right[term] for term in shared_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def chunk_text_by_words(text: str) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - CHUNK_OVERLAP)

    return chunks


def allowed_pdf(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def build_prompt(question: str, contexts: List[dict]) -> str:
    source_blocks = []
    for index, item in enumerate(contexts, start=1):
        source_blocks.append(
            f"Source {index}: {item['filename']}, page {item['page']}\n{item['text']}"
        )

    return f"""You are an HR policy assistant. Answer only from the provided policy excerpts.
If the answer is not present, say that the uploaded documents do not contain enough information.
Keep the answer practical and cite source numbers inline.

Question: {question}

Policy excerpts:
{chr(10).join(source_blocks)}
"""


def answer_locally(question: str, contexts: List[dict]) -> str:
    if not contexts:
        return "I could not find a relevant answer in the uploaded HR documents."

    snippets = []
    for index, item in enumerate(contexts[:2], start=1):
        preview = item["text"]
        if len(preview) > 650:
            preview = preview[:650].rsplit(" ", 1)[0] + "..."
        snippets.append(f"Source {index} says: {preview}")

    return (
        "I found these relevant HR policy excerpts:\n\n"
        + "\n\n".join(snippets)
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    docs = sorted({chunk.filename for chunk in knowledge_base.chunks})
    return jsonify({"documents": docs, "chunks": len(knowledge_base.chunks)})


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Please choose at least one PDF."}), 400

    uploaded = []
    for file in files:
        if not file.filename or not allowed_pdf(file.filename):
            return jsonify({"error": f"{file.filename or 'File'} is not a PDF."}), 400

        filename = secure_filename(file.filename)
        pdf_path = UPLOAD_DIR / filename
        file.save(pdf_path)
        chunks_added = knowledge_base.add_pdf(pdf_path)
        uploaded.append({"filename": filename, "chunks": chunks_added})

    return jsonify({"uploaded": uploaded, "total_chunks": len(knowledge_base.chunks)})


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = normalize_text(data.get("question", ""))

    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if not knowledge_base.ready:
        return jsonify({"error": "Upload at least one HR policy PDF first."}), 400

    contexts = knowledge_base.search(question)
    answer = answer_locally(question, contexts)

    return jsonify({"answer": answer, "sources": contexts})


@app.post("/api/reset")
def reset():
    knowledge_base.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
