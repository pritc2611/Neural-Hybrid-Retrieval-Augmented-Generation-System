FROM python:3.10-slim

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /rag_app

# 🔹 System deps (minimal, needed for PyMuPDF / PIL / Torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 🔹 Copy only requirements first (cache-friendly)
COPY requir.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 🔹 Copy app last
COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag_app:app", "--host", "0.0.0.0", "--port", "8000"]
