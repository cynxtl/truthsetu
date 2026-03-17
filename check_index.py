import json
from pathlib import Path

docs_file = Path('./data/faiss_index/documents.json')

if not docs_file.exists():
    print("documents.json not found!")
else:
    docs = json.loads(docs_file.read_text(encoding='utf-8'))
    print(f"Total docs in FAISS index: {len(docs)}")
    print()
    for d in docs[:5]:
        print(f"Source: {d['source']}")
        print(f"Text:   {d['text'][:100]}")
        print()