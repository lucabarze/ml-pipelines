FROM registry.suse.com/bci/python:3.13
ENV PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir mlflow==2.18.0
COPY fetch_lora.py /fetch_lora.py
ENTRYPOINT ["python", "/fetch_lora.py"]
