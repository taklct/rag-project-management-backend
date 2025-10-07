# app.py
# pip install fastapi uvicorn "openai>=1.40.0" pandas openpyxl PyPDF2 python-docx
import os
import sys
import math
import time
import csv
import pickle
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AzureOpenAI

from project_dashboard import load_project_tasks, router as project_dashboard_router

# ===================== Config =====================
try:
    import config
except Exception:
    config = None

ENDPOINT_URL = os.getenv("ENDPOINT_URL") or getattr(config, "ENDPOINT_URL", None)
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME") or getattr(config, "DEPLOYMENT_NAME", None)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or getattr(config, "AZURE_OPENAI_API_KEY", None)
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME") or getattr(config, "EMBEDDING_DEPLOYMENT_NAME", None)

if not ENDPOINT_URL or not DEPLOYMENT_NAME or not AZURE_OPENAI_API_KEY or not EMBEDDING_DEPLOYMENT_NAME:
    print("Please set ENDPOINT_URL, DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, EMBEDDING_DEPLOYMENT_NAME (env or config.py).")
    sys.exit(1)

client = AzureOpenAI(
    azure_endpoint=ENDPOINT_URL,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2025-01-01-preview",
)

SOURCE_DIR = "./data_sources"
DEFAULT_TOP_K = 3
LOG_DIR = "./logs"
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "query_logs.csv")

# ===================== I/O helpers =====================
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    try:
        import PyPDF2
    except ImportError:
        raise RuntimeError("PyPDF2 not installed. Run: pip install PyPDF2")
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i in range(len(reader.pages)):
            try:
                text.append(reader.pages[i].extract_text() or "")
            except Exception:
                text.append("")
    return "\n".join(text)

def load_document(path: str) -> str:
    p = path.lower()
    if p.endswith((".txt", ".md")):
        return read_text_file(path)
    elif p.endswith(".pdf"):
        return read_pdf(path)
    elif p.endswith((".xlsx", ".xls")):
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for Excel. Run: pip install pandas openpyxl xlrd")
        df_map = pd.read_excel(path, sheet_name=None)
        texts = []
        for sheet, sdf in df_map.items():
            head = f"--- Sheet: {sheet} ---"
            sdf = sdf.fillna("")
            rows = sdf.to_dict(orient="records")
            row_lines = [" | ".join(f"{k}={v}" for k, v in r.items()) for r in rows]
            texts.append(head + "\n" + "\n".join(row_lines))
        return "\n\n".join(texts)
    elif p.endswith(".csv"):
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for CSV. Run: pip install pandas")
        df = pd.read_csv(path).fillna("")
        rows = df.to_dict(orient="records")
        return "\n".join(" | ".join(f"{k}={v}" for k, v in r.items()) for r in rows)
    elif p.endswith(".docx"):
        try:
            import docx
        except ImportError:
            raise RuntimeError("python-docx required for Word files. Run: pip install python-docx")
        doc = docx.Document(path)
        paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras)
    else:
        try:
            return read_text_file(path)
        except Exception:
            raise RuntimeError(f"Unsupported file type: {path}")

# ===================== Chunking & embeddings =====================
def chunk_text(text: str, max_chars: int = 1500, overlap: int = 150) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = start + max_chars
                    chunks.append(p[start:end])
                    start = max(0, end - overlap)
            else:
                buf = p
    if buf:
        chunks.append(buf)

    if overlap <= 0:
        return chunks

    overlapped: List[str] = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            prev = chunks[i - 1]
            tail = prev[-overlap:]
            overlapped.append((tail + "\n" + c).strip())
    return overlapped

def embed_texts(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = client.embeddings.create(model=EMBEDDING_DEPLOYMENT_NAME, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# ===================== On-disk index (PKL next to each XLSX) =====================
@dataclass
class IndexFile:
    pkl_path: str
    chunks: List[str]
    embeddings: List[List[float]]
    signature: str

def file_signature(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    st = os.stat(path)
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()

def pkl_path_for(doc_path: str) -> str:
    base = os.path.basename(doc_path)
    return os.path.join(os.path.dirname(doc_path), f"{base}.{EMBEDDING_DEPLOYMENT_NAME}.idx.pkl")

def load_index_if_valid(doc_path: str) -> Optional[IndexFile]:
    pklp = pkl_path_for(doc_path)
    if not os.path.exists(pklp):
        return None
    try:
        with open(pklp, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            return None
        if obj.get("embedding_model") != EMBEDDING_DEPLOYMENT_NAME:
            return None
        if obj.get("doc_signature") != file_signature(doc_path):
            return None
        return IndexFile(
            pkl_path=pklp,
            chunks=obj["chunks"],
            embeddings=obj["embeddings"],
            signature=obj["doc_signature"],
        )
    except Exception:
        return None

def build_index(doc_path: str) -> IndexFile:
    raw = load_document(doc_path)
    chunks = chunk_text(raw, 1500, 150)
    if not chunks:
        raise RuntimeError(f"No text extracted from {doc_path}")
    embs = embed_texts(chunks)
    sig = file_signature(doc_path)
    payload = {
        "doc_signature": sig,
        "embedding_model": EMBEDDING_DEPLOYMENT_NAME,
        "chunks": chunks,
        "embeddings": embs,
    }
    pklp = pkl_path_for(doc_path)
    with open(pklp, "wb") as f:
        pickle.dump(payload, f)
    return IndexFile(pkl_path=pklp, chunks=chunks, embeddings=embs, signature=sig)

def ensure_index(doc_path: str, rebuild: bool = False) -> IndexFile:
    if not rebuild:
        idx = load_index_if_valid(doc_path)
        if idx:
            return idx
    return build_index(doc_path)

def list_xlsx(source_dir: str) -> List[str]:
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    out = []
    for name in os.listdir(source_dir):
        if name.lower().endswith((".xlsx", ".xls")):
            out.append(os.path.join(source_dir, name))
    return sorted(out)

# ===================== Retrieval & LLM =====================
def retrieve_top_k(question: str, all_chunks: List[str], all_embs: List[List[float]], k: int) -> List[Tuple[int, float]]:
    q_emb = client.embeddings.create(model=EMBEDDING_DEPLOYMENT_NAME, input=[question]).data[0].embedding
    sims = [(i, cosine_similarity(q_emb, emb)) for i, emb in enumerate(all_embs)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

def build_messages(retrieved_chunks: List[str], question: str) -> List[Dict[str, str]]:
    context_block = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(retrieved_chunks)])
    system_text = (
        "You are a helpful AI assistant. Use ONLY the provided context to answer the question. "
        "If the answer is not in the context, say you don't know and suggest what would help."
    )
    user_text = (
        "Here is the context from the document(s):\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly. If relevant, quote short snippets."
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

def ask_llm(messages: List[Dict[str, str]], temperature: float = 1, max_completion_tokens: int = 10000) -> Tuple[str, Dict[str, Optional[int]], float]:
    t0 = time.time()
    completion = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        max_completion_tokens=max_completion_tokens,     # correct param for Chat Completions API
        temperature=temperature,
    )
    dt = time.time() - t0
    msg = completion.choices[0].message
    usage = getattr(completion, "usage", None)
    tokens = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    } if usage else {}
    return (msg.content or "").strip(), tokens, dt

# ===================== Logging =====================
def log_query(row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "question",
                "top_k",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "duration_sec",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ===================== FastAPI app =====================
app = FastAPI(
    title="RAG Project Management Backend",
    description=(
        "Backend APIs for building retrieval-augmented generation (RAG) indexes "
        "and querying project management insights. Use the `/docs` endpoint to "
        "explore the automatically generated Swagger UI."
    ),
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Indexing",
            "description": "Manage retrieval indexes generated from uploaded workbooks.",
        },
        {
            "name": "Q&A",
            "description": "Ask questions over the indexed project documents using RAG.",
        },
        {
            "name": "Project Dashboard",
            "description": "Access metrics and task listings extracted from project workbooks.",
        },
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(project_dashboard_router)

class BuildBody(BaseModel):
    """Request payload for `/build` to trigger index creation."""

    rebuild: bool = Field(
        False,
        description=(
            "Recreate indexes even if cached embeddings already exist for the source "
            "workbooks."
        ),
    )


class IndexedFileSummary(BaseModel):
    """Summary of an indexed workbook and its embedding cache."""

    xlsx: str = Field(..., description="Workbook filename that was processed.")
    pkl: str = Field(..., description="Generated embedding cache filename.")
    chunks: int = Field(..., description="Number of text chunks extracted from the workbook.")
    signature: str = Field(
        ..., description="Short hash of the workbook contents used for cache validation."
    )


class ProjectDashboardStatus(BaseModel):
    """Status information about the project dashboard cache."""

    ok: bool = Field(..., description="Whether project dashboard data is available.")
    count: Optional[int] = Field(
        None, description="Number of cached project tasks when available."
    )
    error: Optional[str] = Field(
        None, description="Details when the dashboard data could not be loaded."
    )


class BuildResponse(BaseModel):
    """Response returned by the `/build` endpoint."""

    ok: Literal[True] = Field(True, description="Whether the build completed successfully.")
    built: List[IndexedFileSummary] = Field(
        default_factory=list,
        description="Details for each workbook that was indexed.",
    )
    message: Optional[str] = Field(
        None, description="Helpful message when no workbooks were available to index."
    )
    project_dashboard: ProjectDashboardStatus = Field(
        ..., description="Status of the project dashboard cache after the build runs."
    )


class AskBody(BaseModel):
    """Request payload for `/ask` to run a RAG-powered query."""

    question: str = Field(..., description="Natural language question to ask over the data.")
    top_k: Optional[int] = Field(
        None,
        description="Number of most similar chunks to retrieve from the index (defaults to 3).",
        ge=1,
    )
    temperature: Optional[float] = Field(
        None,
        description=(
            "Sampling temperature for the language model. Leave unset to use the "
            "default configuration."
        ),
        ge=0.0,
    )


class TokenUsage(BaseModel):
    """Token accounting metadata returned by Azure OpenAI."""

    prompt_tokens: Optional[int] = Field(
        None, description="Tokens consumed by the prompt portion of the request."
    )
    completion_tokens: Optional[int] = Field(
        None, description="Tokens generated in the completion response."
    )
    total_tokens: Optional[int] = Field(
        None, description="Total tokens counted for the request.")


class AskSuccessResponse(BaseModel):
    """Successful response for the `/ask` endpoint."""

    ok: Literal[True] = Field(True, description="Indicates the question was processed.")
    answer: str = Field(..., description="Generated answer from the language model.")
    usage: Optional[TokenUsage] = Field(
        None, description="Token usage metrics reported by the language model."
    )
    duration_sec: float = Field(..., description="Time taken to build the response in seconds.")
    top_scores: List[float] = Field(
        ..., description="Similarity scores for the retrieved context chunks."
    )
    indexed_files: List[str] = Field(
        ..., description="Workbook filenames that contributed to the retrieval context."
    )


class AskErrorResponse(BaseModel):
    """Error payload returned when the `/ask` endpoint cannot run."""

    ok: Literal[False] = Field(False, description="Indicates the question could not be run.")
    error: str = Field(..., description="Reason why the request failed.")

@app.post(
    "/build",
    response_model=BuildResponse,
    tags=["Indexing"],
    summary="Generate embeddings for available workbooks",
    response_description="Details about the index build and project dashboard cache.",
)
def build_indexes(body: BuildBody) -> BuildResponse:
    xlsx_list = list_xlsx(SOURCE_DIR)
    if not xlsx_list:
        response: Dict[str, Any] = {
            "ok": True,
            "built": [],
            "message": "No .xlsx/.xls files found in ./data_sources.",
        }
        try:
            dashboard = load_project_tasks()
        except FileNotFoundError:
            dashboard = {"ok": False, "error": "Project tasks workbook not found."}
        except Exception as exc:  # pragma: no cover - defensive
            dashboard = {"ok": False, "error": str(exc)}
        response["project_dashboard"] = ProjectDashboardStatus(**dashboard)
        return BuildResponse(**response)

    built = []
    for path in xlsx_list:
        idx = ensure_index(path, rebuild=body.rebuild)
        built.append({
            "xlsx": os.path.basename(path),
            "pkl": os.path.basename(idx.pkl_path),
            "chunks": len(idx.chunks),
            "signature": idx.signature[:16]
        })
    try:
        dashboard = load_project_tasks()
    except FileNotFoundError:
        dashboard = {"ok": False, "error": "Project tasks workbook not found."}
    except Exception as exc:  # pragma: no cover - defensive
        dashboard = {"ok": False, "error": str(exc)}

    return BuildResponse(built=built, project_dashboard=ProjectDashboardStatus(**dashboard))

@app.post(
    "/ask",
    response_model=AskSuccessResponse | AskErrorResponse,
    tags=["Q&A"],
    summary="Ask a question using retrieval augmented generation",
    response_description="Generated answer or an error describing why the request failed.",
)
def ask_question(body: AskBody) -> AskSuccessResponse | AskErrorResponse:
    top_k = body.top_k or DEFAULT_TOP_K
    temperature = 1 if body.temperature is None else body.temperature

    xlsx_list = list_xlsx(SOURCE_DIR)
    if not xlsx_list:
        return AskErrorResponse(
            error="No .xlsx/.xls files in ./data_sources. Build first via /build."
        )

    all_chunks: List[str] = []
    all_embs: List[List[float]] = []
    for path in xlsx_list:
        idx = ensure_index(path, rebuild=False)
        all_chunks.extend(idx.chunks)
        all_embs.extend(idx.embeddings)

    top = retrieve_top_k(body.question, all_chunks, all_embs, k=top_k)
    retrieved = [all_chunks[i] for i, _ in top]
    top_scores = [round(s, 4) for _, s in top]

    messages = build_messages(retrieved, body.question)
    answer, usage, duration = ask_llm(messages, temperature=temperature, max_completion_tokens=10000)

    now_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_query({
        "timestamp": now_iso,
        "question": body.question,
        "top_k": top_k,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "duration_sec": round(duration, 3),
    })

    token_usage = TokenUsage(**usage) if usage else None

    return AskSuccessResponse(
        answer=answer,
        usage=token_usage,
        duration_sec=round(duration, 3),
        top_scores=top_scores,
        indexed_files=[os.path.basename(p) for p in xlsx_list],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
