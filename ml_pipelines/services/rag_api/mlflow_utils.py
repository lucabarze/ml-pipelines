# mlflow_utils.py
import json
import os
import time
from contextlib import contextmanager
from typing import Dict, Optional

try:
    import mlflow
except Exception:  # MLflow opzionale
    mlflow = None

_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "default")
_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
_RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "chat")
_TAGS_RAW = os.getenv("MLFLOW_TAGS", "")

def _parse_tags(s: str) -> Dict[str, str]:
    if not s:
        return {}
    try:
        d = json.loads(s)
        return {str(k): str(v) for k,v in d.items()}
    except Exception:
        return {}

MLFLOW_TAGS = _parse_tags(_TAGS_RAW)

def mlflow_init():
    """Call at startup (safe if MLflow missing)."""
    if mlflow is None:
        return
    if _TRACKING_URI:
        mlflow.set_tracking_uri(_TRACKING_URI)
    try:
        mlflow.set_experiment(_EXPERIMENT)
    except Exception:
        pass

@contextmanager
def mlflow_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, nested: bool = True):
    """Context manager per un run per-request."""
    if mlflow is None:
        yield None
        return
    rn = run_name or _RUN_NAME
    tg = dict(MLFLOW_TAGS)
    if tags:
        tg.update({str(k): str(v) for k,v in tags.items()})
    with mlflow.start_run(run_name=rn, tags=tg, nested=nested):
        yield mlflow

def log_params_safe(d: Dict):
    if mlflow is None:
        return
    try:
        # MLflow non accetta dizionari annidati o valori lunghi
        flat = {str(k): str(v) for k, v in d.items()}
        mlflow.log_params(flat)
    except Exception:
        pass

def log_metrics_safe(d: Dict[str, float], step: Optional[int] = None):
    if mlflow is None:
        return
    try:
        mlflow.log_metrics({str(k): float(v) for k, v in d.items()}, step=step)
    except Exception:
        pass

class Timer:
    """Timer semplice per fasi della request."""
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0
        self.measures = {}

    def lap(self, name: str):
        now = time.time()
        self.measures[name] = now - self.last
        self.last = now

    def total(self) -> float:
        return time.time() - self.t0

    def as_ms(self) -> Dict[str, float]:
        return {k: v * 1000.0 for k, v in self.measures.items()}
