FROM registry.suse.com/bci/python:3.12

WORKDIR /app

RUN rm -f /usr/lib/python3.12/site-packages/distutils-precedence.pth || true

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "setuptools==70.2.0" wheel

COPY constraints.txt /app/constraints.txt
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt -c /app/constraints.txt

ENV XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/hf \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    HUGGINGFACE_HUB_CACHE=/app/.cache/hub \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/st

RUN mkdir -p /app/.cache/hf /app/.cache/transformers /app/.cache/hub /app/.cache/st && \
    chmod -R 777 /app/.cache 

COPY app.py /app/app.py
COPY mlflow_utils.py /app/mlflow_utils.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
