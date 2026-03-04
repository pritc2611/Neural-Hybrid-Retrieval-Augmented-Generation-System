"""
ingestion.py — Document extraction, parallel embedding, Pinecone upsert.
"""
import os
import re
import time
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from langsmith import traceable

def _get_cfg():
    import utility.config as cfg
    return cfg


# =============================================================================
# EXTRACTION  (CPU-bound — run in ProcessPoolExecutor)
# =============================================================================

def _extract_pdf_worker(pdf_bytes: bytes, filename: str) -> List[Dict]:
    import pymupdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import utility.config as cfg
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    doc  = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    full = "\n\n".join(p.get_text() for p in doc if p.get_text().strip())
    return [{"text": c, "source": filename, "chunk_id": i}
            for i, c in enumerate(splitter.split_text(full))]


def _extract_text_worker(text: str, filename: str) -> List[Dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import utility.config as cfg
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [{"text": c, "source": filename, "chunk_id": i}
            for i, c in enumerate(splitter.split_text(text))]


# =============================================================================
# PARALLEL EMBEDDING
# =============================================================================
async def _embed_chunks_parallel(chunks: List[Dict]) -> List[Dict]:
    cfg     = _get_cfg()
    texts   = [c["text"] for c in chunks]
    batches = [texts[i: i + cfg.EMBEDDING_BATCH_SIZE]
               for i in range(0, len(texts), cfg.EMBEDDING_BATCH_SIZE)]
    t0 = time.time()
    results = await asyncio.gather(
        *[cfg.embedding_client.embed(texts=b) for b in batches],
        return_exceptions=True,
    )
    print(f"  🔢 {len(texts)} chunks embedded in {time.time()-t0:.2f}s ({len(batches)} batches)")

    all_embeddings: List[List[float]] = []
    for r in results:
        if isinstance(r, Exception):
            for _ in range(cfg.EMBEDDING_BATCH_SIZE):
                all_embeddings.append([0.0] * 768)
        elif isinstance(r[0], list):
            all_embeddings.extend(r)
        else:
            all_embeddings.append(r)

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb
    return chunks


# =============================================================================
# PINECONE UPSERT
# =============================================================================
@traceable(name="upserting to pinecone")
async def _upsert_to_pinecone(chunks: List[Dict], namespace: str):
    cfg     = _get_cfg()
    vectors = [
        {
            "id":       f"{namespace}-{c['source']}-{c['chunk_id']}",
            "values":   c["embedding"],
            "metadata": {"text": c["text"], "source": c["source"], "chunk_id": c["chunk_id"]},
        }
        for c in chunks
    ]
    sub_batches = [vectors[i: i + 100] for i in range(0, len(vectors), 100)]
    await asyncio.gather(
        *[asyncio.to_thread(cfg.pc_index.upsert, vectors=sb, namespace=namespace)
          for sb in sub_batches]
    )


# =============================================================================
# PUBLIC UPLOAD FUNCTIONS
# =============================================================================
@traceable(name="extracting and storing")
async def extract_and_store(
    filename: str, file_obj, target_namespace: Optional[str] = None,
) -> List[Dict]:
    from utility.retriver import _update_bm25
    cfg  = _get_cfg()
    t0   = time.time()
    loop = asyncio.get_event_loop()
    print(f"📄 Processing: {filename}")

    if filename.lower().endswith(".pdf"):
        pdf_bytes = file_obj.file.read()
        chunks = await loop.run_in_executor(
            cfg._process_pool, _extract_pdf_worker, pdf_bytes, filename,
        )
    elif filename.lower().endswith((".txt", ".md")):
        raw_text = file_obj.file.read().decode("utf-8")
        chunks = await loop.run_in_executor(
            cfg._process_pool, _extract_text_worker, raw_text, filename,
        )
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    print(f"  ✂️  {len(chunks)} chunks extracted ({time.time()-t0:.2f}s)")
    chunks = await _embed_chunks_parallel(chunks)

    if target_namespace:
        namespace = target_namespace
    else:
        name      = os.path.splitext(filename)[0]
        namespace = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:63]
        cfg.current_namespace = namespace

    await _upsert_to_pinecone(chunks, namespace)
    await loop.run_in_executor(
        cfg._thread_pool, _update_bm25, namespace,
        [c["text"] for c in chunks],
        [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks],
    )
    print(f"✅ {len(chunks)} chunks → ns='{namespace}' ({time.time()-t0:.2f}s)")
    return chunks


async def extract_and_store_multiple(files: List[Tuple[str, Any]]) -> Dict[str, Any]:
    cfg        = _get_cfg()
    names      = [re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.splitext(fn)[0])[:20] for fn, _ in files]
    combined_ns = "__".join(names)[:63]
    cfg.current_namespace = combined_ns
    print(f"📦 Multi-upload → combined namespace: '{combined_ns}'")
    tasks   = [asyncio.create_task(extract_and_store(fn, fo, target_namespace=combined_ns)) for fn, fo in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {
        fn: {"status": "ok", "chunks": len(r)} if not isinstance(r, Exception)
            else {"status": "error", "error": str(r)}
        for (fn, _), r in zip(files, results)
    }