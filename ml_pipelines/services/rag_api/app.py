import os
import time
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer 

import logging
from mlflow_utils import mlflow_init, mlflow_run, log_params_safe, log_metrics_safe, Timer

# -------------------------
# Config via env
# -------------------------
VLLM_URL = os.getenv("VLLM_URL", "http://vllm.apps.eni.lajoie.de/v1/chat/completions")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "none")
VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "phi3-mini") 

MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus.milvus.svc.cluster.local")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks")
MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "embedding")
MILVUS_TEXT_FIELD = os.getenv("MILVUS_TEXT_FIELD", "text")
TOP_K = int(os.getenv("TOP_K", "3"))

EMB_MODEL_NAME = os.getenv("EMB_MODEL", "intfloat/e5-small-v2")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Ground your answers ONLY on the provided context. "
    "If the answer is not in the context, say you don't know."
)

CTX_MAX_TOKENS = int(os.getenv("CTX_MAX_TOKENS", "1000"))   
CTX_CHUNK_TOKENS = int(os.getenv("CTX_CHUNK_TOKENS", "220"))
TOKENIZER_ID = os.getenv("TOKENIZER_ID", "microsoft/Phi-3-mini-4k-instruct")


# -------------------------
# Init
# -------------------------
app = FastAPI(title="RAG API (OpenAI-compatible)")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-api")

collection: Optional[Collection] = None
embedder: Optional[SentenceTransformer] = None
tok = None

@app.on_event("startup")
def _startup():
    global collection, embedder, tok
    mlflow_init()
    for i in range(10):
        try:
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
            c = Collection(MILVUS_COLLECTION)
            c.load()
            collection = c
            break
        except Exception as e:
            time.sleep(2)
    if collection is None:
        raise RuntimeError("Milvus not ready")
    cache_dir = os.getenv("MODEL_CACHE", "/app/.cache/models")
    os.makedirs(cache_dir, exist_ok=True)
    embedder = SentenceTransformer(EMB_MODEL_NAME, cache_folder=cache_dir)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, cache_dir=cache_dir)
    _ = embedder.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)

def _embed(texts: List[str]):
    texts = [f"query: {t}" for t in texts]
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

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

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "owner"

@app.get("/v1/models")
def list_models():
    return {"object":"list", "data":[ModelInfo(id=VLLM_MODEL_ID)] }

# -------------------------
# Utils
# -------------------------
def _last_user_question(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

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


def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens: return text
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)

def _build_prompt(system_prompt: str, user_question: str, contexts: List[str]) -> List[Dict[str, str]]:
    trimmed, used = [], 0
    for c in contexts:
        tc = _truncate_by_tokens(c, CTX_CHUNK_TOKENS)
        tlen = len(tok.encode(tc, add_special_tokens=False))
        if used + tlen > CTX_MAX_TOKENS:
            break
        trimmed.append(tc); used += tlen
        context_block = "\n\n".join(f"- {c}" for c in trimmed) if trimmed else "N/A"
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

    question = _last_user_question([m.model_dump() for m in req.messages])
    if not question:
        raise HTTPException(status_code=400, detail="no user message provided")

    t = Timer()

    with mlflow_run(tags={"route": "/v1/chat/completions"}) as mlf:
        log_params_safe({
            "model_req": req.model or "",
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "top_p": req.top_p,
            "TOP_K": TOP_K,
        })

        contexts = _retrieve(question)
        t.lap("retrieval")

        messages = _build_prompt(SYSTEM_PROMPT, question, contexts)
        ctx_chars = sum(len(c) for c in contexts)
        log_metrics_safe({"ctx_chunks": len(contexts), "ctx_chars": ctx_chars})

        payload = {
            "model": req.model or VLLM_MODEL_ID,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "top_p": req.top_p,
            "stop": req.stop,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
        try:
            r = requests.post(VLLM_URL, json=payload, headers=headers, timeout=120)
        except Exception as e:
            log.error("vLLM unreachable: %s", e)
            log_metrics_safe({"error_vllm_unreachable": 1})
            raise HTTPException(status_code=502, detail=f"vLLM unreachable: {e}")

        if r.status_code >= 300:
            log.error("vLLM error: %s %s", r.status_code, r.text[:200])
            log_metrics_safe({"error_vllm_http": r.status_code})
            raise HTTPException(status_code=502, detail=f"vLLM error: {r.status_code} {r.text}")

        data = r.json()
        t.lap("vllm")

        try:
            choice = data["choices"][0]["message"]["content"]
            model_name = data.get("model", "vllm")
            created = data.get("created", int(time.time()))
        except Exception:
            log.error("invalid response from vLLM: %s", str(data)[:200])
            log_metrics_safe({"error_invalid_vllm_resp": 1})
            raise HTTPException(status_code=500, detail="invalid response from vLLM")

        usage = Usage(
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
        )

        per_phase = t.as_ms()
        per_phase["total_ms"] = t.total() * 1000.0
        per_phase.update({
            "usage_prompt_tokens": usage.prompt_tokens,
            "usage_completion_tokens": usage.completion_tokens,
            "usage_total_tokens": usage.total_tokens,
        })
        log_metrics_safe(per_phase)

        log.info(
            "retr=%.0fms vllm=%.2fs total=%.2fs ctx_chunks=%d ctx_chars=%d prompt=%d compl=%d",
            per_phase.get("retrieval", 0),
            per_phase.get("vllm", 0)/1000.0,
            per_phase["total_ms"]/1000.0,
            len(contexts),
            ctx_chars,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        return ChatCompletionResponse(
            id=data.get("id", "rag-completion-1"),
            created=created,
            model=model_name,
            choices=[Choice(index=0, message=ChoiceMessage(role="assistant", content=choice))],
            usage=usage,
        )
