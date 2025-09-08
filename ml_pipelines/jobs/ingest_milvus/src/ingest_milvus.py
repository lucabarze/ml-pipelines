import os, argparse, glob, re, json, tempfile
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# ---------- Markdown → testo ----------
def read_markdown_files(docs_dir: str):
    paths = sorted(glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True))
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            yield p, f.read()

def strip_md(md: str) -> str:
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    md = re.sub(r"`([^`]+)`", r"\1", md)
    md = re.sub(r"!$begin:math:display$[^$end:math:display$]*\]$begin:math:text$[^)]+$end:math:text$", "", md)
    md = re.sub(r"$begin:math:display$[^$end:math:display$]+\]$begin:math:text$([^)]+)$end:math:text$", r"\1", md)
    md = re.sub(r"^#+\s*", "", md, flags=re.M)
    md = re.sub(r"\s+", " ", md).strip()
    return md

def chunk_text(text: str, max_chars=1200, overlap=150):
    parts, i = [], 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        k = text.rfind(".", i, j)
        if k == -1:
            k = text.rfind("\n", i, j)
        if k == -1:
            k = j
        parts.append(text[i:k].strip())
        i = max(k - overlap, i + 1)
    return [p for p in parts if len(p) > 0]

# ---------- Embeddings ----------
def load_encoder(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()
    return tok, mdl, device, mdl.config.hidden_size

@torch.no_grad()
def embed(texts, tok, mdl, device, max_len=512, normalize=True):
    out = []
    for i in range(0, len(texts), 32):
        batch = tok(texts[i:i+32], truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
        vec = mdl(**batch).last_hidden_state[:,0,:]
        v = vec.detach().cpu().numpy().astype("float32")
        out.append(v)
    X = np.concatenate(out, axis=0) if out else np.zeros((0, tok.model_max_length), dtype="float32")
    if normalize and len(X):
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

# ---------- Milvus ----------
def ensure_collection(name: str, dim: int, drop: bool, metric="COSINE"):
    if utility.has_collection(name):
        if drop:
            utility.drop_collection(name)
        else:
            return Collection(name)
    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    ], description="RAG chunks from Markdown")
    col = Collection(name, schema=schema)
    col.create_index("embedding", {"index_type":"HNSW","metric_type":metric,"params":{"M":16,"efConstruction":200}})
    return col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", required=True, help="Cartella con file .md")
    ap.add_argument("--milvus_host", default="milvus.myns.svc.cluster.local")
    ap.add_argument("--milvus_port", default="19530")
    ap.add_argument("--collection", default="rag_chunks")
    ap.add_argument("--emb_model", default="intfloat/e5-small-v2")
    ap.add_argument("--drop_if_exists", action="store_true")
    ap.add_argument("--chunk_chars", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    args = ap.parse_args()

    docs = list(read_markdown_files(args.docs_dir))
    texts, doc_ids, chunk_ids, paths = [], [], [], []
    for path, md in docs:
        base = Path(path).stem
        chunks = chunk_text(strip_md(md), max_chars=args.chunk_chars, overlap=args.chunk_overlap)
        for idx, ch in enumerate(chunks):
            texts.append(ch)
            doc_ids.append(base)
            chunk_ids.append(idx)
            paths.append(path)
    print(f"[INGEST] files: {len(docs)}  chunks: {len(texts)}")

    tok, mdl, device, dim = load_encoder(args.emb_model)
    print(f"[EMB] {args.emb_model} dim={dim} device={device}")

    connections.connect("default", host=args.milvus_host, port=args.milvus_port)
    col = ensure_collection(args.collection, dim, args.drop_if_exists, metric="COSINE")
    col.load()

    B = 256
    total = 0
    for s in range(0, len(texts), B):
        batch_texts = texts[s:s+B]
        vecs = embed(batch_texts, tok, mdl, device)
        mr = col.insert([vecs, doc_ids[s:s+B], chunk_ids[s:s+B], paths[s:s+B], batch_texts])
        total += len(batch_texts)
        if total % 1024 == 0:
            print(f"[INGEST] inserted {total}")
    col.flush()
    print(f"[DONE] inserted total: {total}")

if __name__ == "__main__":
    main()
