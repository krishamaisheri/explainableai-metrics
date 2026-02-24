"""
Build Vector Store — One-time CLI script.

Extracts text from all PDFs in data/, saves as .md files,
chunks them, and stores embeddings in a persistent JSON+numpy store.

Usage:
    python3 build_vector_store.py
"""

import time
from pdf_extractor import extract_all_pdfs
from vector_store import ingest, get_collection_stats, query


def run():
    print("=" * 60)
    print("🔨 PDF EXTRACTION + VECTOR STORE BUILD")
    print("=" * 60)

    # Step 1: Extract text from PDFs
    print("\n📚 Step 1: Extracting text from PDFs…\n")
    start = time.time()
    chunks = extract_all_pdfs()
    extract_time = time.time() - start
    print(f"\n   ⏱️  Extraction took {extract_time:.1f}s")

    if not chunks:
        print("❌ No chunks extracted. Check that data/ contains PDF files.")
        return

    # Step 2: Ingest into vector store
    print("\n🗄️  Step 2: Embedding + ingesting into vector store…\n")
    start = time.time()
    count = ingest(chunks)
    ingest_time = time.time() - start
    print(f"   ⏱️  Ingestion took {ingest_time:.1f}s")

    # Step 3: Summary
    stats = get_collection_stats()
    print("\n" + "=" * 60)
    print("✅ VECTOR STORE READY")
    print(f"   📊 Collection: {stats['collection_name']}")
    print(f"   📄 Documents:  {stats['document_count']}")
    print(f"   💾 Stored at:  {stats['persist_dir']}")
    print("=" * 60)

    # Step 4: Verification query
    print("\n🔍 Verification query: 'disability benefit eligibility'\n")
    results = query("disability benefit eligibility", top_k=3)
    for r in results:
        print(f"   📄 [{r['category']}] {r['source']} (dist: {r['distance']:.4f})")
        print(f"      {r['text'][:120]}…")
        print()

    print("✅ Done. Vector store is ready for the RAG pipeline.")


if __name__ == "__main__":
    run()
