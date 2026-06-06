# SRag — Docker image
# Builds a self-contained SRag HTTP server
#
# Usage:
#   docker build -t srag .
#   docker run -p 8000:8000 srag
#
# With volume for persistent LanceDB + intelligence data:
#   docker run -p 8000:8000 -v $(pwd)/srag_data:/app/srag_data srag

FROM python:3.11-slim

# System deps — needed for Playwright + trafilatura
RUN apt-get update && apt-get install -y \
    wget curl git \
    libglib2.0-0 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install SRag with all optional deps
# In local dev, replace with: COPY . . && pip install -e ".[all]"
RUN pip install --no-cache-dir "srag[all]" uvicorn fastapi

# Pre-download sentence-transformers model so it's cached in the image
# This means the container doesn't need internet on first run
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model cached.')"

# Install Playwright browser (Chromium only — keeps image smaller)
RUN playwright install chromium --with-deps || true

# Persistent data directory — mount this as a volume to keep
# LanceDB sessions, ReputationStore, LexiconStore across restarts
RUN mkdir -p /app/srag_data
ENV SRAG_DATA_DIR=/app/srag_data

# HuggingFace token — pass at runtime via -e HF_TOKEN=xxx
# or bake into a .env file mounted at /app/.env
ENV HF_TOKEN=""

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "srag.server:app", "--host", "0.0.0.0", "--port", "8000"]