FROM python:3.10 as builder
WORKDIR /rag_app
COPY requir.txt .
RUN pip install --no-cache-dir -r requir.txt

FROM python:3.10-slim
WORKDIR /rag_app
COPY . .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "RAGapp:app", "--host", "0.0.0.0", "--port", "8000"]