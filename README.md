
🤖 Multimodal RAG Application  
Ask Questions About PDFs with Text & Images

🔍 Overview

This is a Multimodal Retrieval-Augmented Generation (RAG) system that allows users to upload any Documents type with containing text and images and ask natural language questions to get accurate, document-grounded answers.
It understands both text and images together, making it suitable for real-world documents like research papers, manuals, and reports.

🎯 Why This Project Exists

Traditional PDF Q&A systems fail when:
- The answer is inside a diagram or image
- Text references visual content
- Meaning depends on text + image together

This system solves that using multimodal embeddings and fused vectors.

✨ Key Features

- Upload PDFs containing text and images
- Automatic text and image extraction (page-wise)
- Multimodal embeddings using CLIP Model
- Fused embeddings for better retrieval
- Vector storage using Pinecone
- Accurate, context-aware answers
- Handles image-based and text-based queries


🧠 Core Concept (Simple Explanation)

1. A PDF is uploaded
2. Each page's text and images are extracted
3. Text and images are converted into vectors
4. Vectors are stored in a vector database
5. User queries retrieve the most relevant vectors
6. An LLM generates the final answer using retrieved context

This approach is known as Retrieval-Augmented Generation (RAG).



🏗️ System Architecture

User
 ↓
Upload PDF
 ↓
Text + Image Extraction
 ↓
Embedding Generation (Text, Image, Fused)
 ↓
Pinecone Vector Database
 ↓
Retriever
 ↓
LLM
 ↓
Final Answer


🔄 Retrieval Logic

1. User submits a query
2. Query embedding is generated
3. Fused vectors are searched first
4. Relevant Documents(chunks) are retrieved
5. Prepare Documents and inject in Promptemplate
6. Prompt is sent to LLM
7. Final answer is generated


📈 Future Improvements

- Image captioning
- Better reranking
- Query rewriting
- Feedback learning loop
