FROM docker.io/rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf_cache TRANSFORMERS_NO_ADVISORY_WARNINGS=1

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-deps -r /app/requirements.txt
