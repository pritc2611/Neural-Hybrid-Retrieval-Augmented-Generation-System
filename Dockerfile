FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100

WORKDIR /rag_app

COPY requir.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requir.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag_app:app", "--host", "0.0.0.0", "--port", "8000"]
