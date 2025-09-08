import os
import time
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# -------------------------
# Config via env
# -------------------------
VLLM_URL = os.getenv("VLLM_URL", "http://vllm-router.default.svc.cluster.local:80/v1/chat/completions")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "none")  # se il router non verifica, va bene un dummy

MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus.default.svc.cluster.local")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks")
MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "embedding")
MILVUS_TEXT_FIELD = os.getenv("MILVUS_TEXT_FIELD", "text")
TOP_K = int(os.getenv("TOP_K", "4"))

EMB_MODEL_NAME = os.getenv("EMB_MODEL", "intfloat/e5-small-v2")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Ground your answers ONLY on the provided context. "
    "If the answer is not in the context, say you don't know."
)

# -------------------------
# Init
# -------------------------
app = FastAPI(title="RAG API (OpenAI-compatible)")

# Milvus connection (single global)
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(MILVUS_COLLECTION)
collection.load()

# Embedding model
embedder = SentenceTransformer(EMB_MODEL_NAME)

# -------------------------
# Schemi OpenAI-compatible (minimal)
# -------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class ChoiceMessage(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)

# -------------------------
# Utils
# -------------------------
def _last_user_question(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def _embed(texts: List[str]):
    # e5 expects "query: ..." prefix for queries
    texts = [f"query: {t}" for t in texts]
    vecs = embedder.encode(texts, normalize_embeddings=True).tolist()
    return vecs

def _retrieve(ctx_query: str) -> List[str]:
    vec = _embed([ctx_query])[0]
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    hits = collection.search(
        data=[vec],
        anns_field=MILVUS_VECTOR_FIELD,
        param=search_params,
        limit=TOP_K,
        output_fields=[MILVUS_TEXT_FIELD],
    )
    contexts: List[str] = []
    if hits and len(hits) > 0:
        for h in hits[0]:
            entity = h.entity
            txt = entity.get(MILVUS_TEXT_FIELD) if entity else None
            if isinstance(txt, bytes):
                txt = txt.decode("utf-8", errors="ignore")
            if txt:
                contexts.append(str(txt))
    return contexts

def _build_prompt(system_prompt: str, user_question: str, contexts: List[str]) -> List[Dict[str, str]]:
    context_block = "\n\n".join([f"- {c}" for c in contexts]) if contexts else "N/A"
    sys = system_prompt + "\n\nContext:\n" + context_block
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_question},
    ]

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# OpenAI-compatible endpoint
# -------------------------
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=false only in this minimal API")

    # 1) Prendi l'ultima domanda utente
    question = _last_user_question([m.model_dump() for m in req.messages])
    if not question:
        raise HTTPException(status_code=400, detail="no user message provided")

    # 2) Retrieval
    contexts = _retrieve(question)

    # 3) Costruisci messaggi per vLLM
    messages = _build_prompt(SYSTEM_PROMPT, question, contexts)

    # 4) Chiama vLLM (OpenAI-compatible)
    payload = {
        "model": req.model or "vllm",   # label del router; vLLM non sempre lo usa
        "messages": messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
        "stop": req.stop,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    t0 = time.time()
    try:
        r = requests.post(VLLM_URL, json=payload, headers=headers, timeout=120)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vLLM unreachable: {e}")
    latency = time.time() - t0

    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"vLLM error: {r.status_code} {r.text}")

    data = r.json()

    # 5) Rispondi in formato OpenAI (pass-through dei campi principali)
    # data è già OpenAI-like (vLLM), ma garantiamo struttura minima
    try:
        choice = data["choices"][0]["message"]["content"]
        model_name = data.get("model", "vllm")
        created = data.get("created", int(time.time()))
    except Exception:
        raise HTTPException(status_code=500, detail="invalid response from vLLM")

    # Stima usage se vLLM non lo fornisce (qui 0 per semplicità)
    usage = Usage(
        prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
        completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
        total_tokens=data.get("usage", {}).get("total_tokens", 0),
    )

    return ChatCompletionResponse(
        id=data.get("id", "rag-completion-1"),
        created=created,
        model=model_name,
        choices=[Choice(index=0, message=ChoiceMessage(role="assistant", content=choice))],
        usage=usage,
    )
