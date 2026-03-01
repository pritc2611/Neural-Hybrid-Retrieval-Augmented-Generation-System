FROM python:3.10-slim AS builder

# Avoid Python bytecode and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build tools temporarily
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY Multi-Modal-RAG/requirements.txt .

# Install dependencies into /install
RUN pip install --prefix=/dependencies -r requirements.txt \
 && apt-get purge -y --auto-remove build-essential


FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /dependencies /usr/local

# Add packages to PATH
ENV PATH=/usr/local/bin:$PATH

# 🔹 Copy app last
COPY  Multi-Modal-RAG/static/  .
COPY Multi-Modal-RAG/templates .
COPY Multi-Modal-RAG/utility .
COPY Multi-Modal-RAG/app/app.py .
 COPY Multi-Modal-RAG/.env .

EXPOSE 7000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7000"]
