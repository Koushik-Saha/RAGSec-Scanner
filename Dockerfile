# ragsec-scanner: API + demo page.
#   docker build -t ragsec-scanner .
#   docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY -e RAGSEC_MODEL ragsec-scanner
# The API key and model come from the environment at runtime and are never
# baked into the image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    RAGSEC_DATA_DIR=/app/data \
    RAGSEC_CHROMA_DIR=/app/chroma \
    RAGSEC_REPORTS_DIR=/app/reports

WORKDIR /app

# CPU-only PyTorch: the embedding model runs on CPU, and the default x86_64
# wheel bundles CUDA libraries that would add gigabytes to the image.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && rm -rf build src/*.egg-info

# Embedding model: baked in at build time rather than downloaded at startup.
# Tradeoff: the image is ~90 MB larger and the model version is frozen at
# build time, but containers start without network access to Hugging Face,
# start faster, and every container runs the exact same model.
RUN python -c "from ragsec.index import SentenceTransformerEmbedder; SentenceTransformerEmbedder().embed(['warm-up'])"
# Never reach out to Hugging Face at runtime; use only the baked-in copy.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Synthetic demo corpus. Indexed into RAGSEC_CHROMA_DIR on first startup.
COPY data ./data

RUN useradd --create-home --uid 10001 ragsec \
    && mkdir -p /app/chroma /app/reports \
    && chown -R ragsec:ragsec /app/chroma /app/reports
USER ragsec

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "ragsec.api:app", "--host", "0.0.0.0", "--port", "8000"]
