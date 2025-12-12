FROM python:3.10-slim

# Set working directory
WORKDIR /rag_app

# Copy and install dependencies first
COPY requir.txt .
RUN pip install --no-cache-dir -r requir.txt

# Copy application source code
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "RAGapp:app", "--host", "0.0.0.0", "--port", "8000"]
