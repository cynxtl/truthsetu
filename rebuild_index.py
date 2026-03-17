import json
import sys
import feedparser
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup

# Load .env manually
from dotenv import load_dotenv
load_dotenv()

SOURCES = [
    ("https://pib.gov.in/RssMain.aspx?ModID=6&Lang=1",                    "pib.gov.in",        0.95),
    ("https://mausam.imd.gov.in/imd_latest/contents/rss-feed.php",        "imd.gov.in",        0.95),
    ("https://ndma.gov.in/RSS/ndma.xml",                                   "ndma.gov.in",       0.95),
    ("https://www.who.int/rss-feeds/news-releases.xml",                    "who.int",           0.90),
    ("https://www.thehindu.com/news/feeder/default.rss",                   "thehindu.com",      0.90),
    ("https://indianexpress.com/feed/",                                    "indianexpress.com", 0.90),
    ("https://feeds.feedburner.com/ndtvnews-top-stories",                  "ndtv.com",          0.88),
    ("https://www.altnews.in/feed/",                                       "altnews.in",        0.95),
    ("https://www.boomlive.in/feed",                                       "boomlive.in",       0.95),
    ("https://timesofindia.indiatimes.com/rssfeedstopstories.cms",         "timesofindia.com",  0.80),
    ("https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",    "hindustantimes.com",0.80),
]

RUMOUR_PHRASES = [
    "viral message claims", "whatsapp forward",
    "rumours suggest", "unverified reports",
    "social media claims", "it is being claimed",
]

TIER_MAP = {
    "pib.gov.in":        "government",
    "imd.gov.in":        "government",
    "ndma.gov.in":       "government",
    "who.int":           "government",
    "altnews.in":        "fact_check",
    "boomlive.in":       "fact_check",
    "thehindu.com":      "tier1_news",
    "indianexpress.com": "tier1_news",
    "ndtv.com":          "tier1_news",
    "timesofindia.com":  "tier2_news",
    "hindustantimes.com":"tier2_news",
}

print("Fetching RSS feeds...")
docs = []

for url, key, weight in SOURCES:
    try:
        feed = feedparser.parse(url)
        count = 0
        for entry in feed.entries[:40]:
            title   = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            link    = entry.get("link", "")
            date    = entry.get("published", "")

            if not title:
                continue

            # Strip HTML
            summary = BeautifulSoup(
                summary, "html.parser"
            ).get_text(separator=" ").strip()[:500]

            text = f"{title}. {summary}".strip()

            # Skip rumour-reporting chunks
            if any(p in text.lower() for p in RUMOUR_PHRASES):
                continue

            docs.append({
                "text":   text,
                "source": key,
                "url":    link,
                "weight": weight,
                "tier":   TIER_MAP.get(key, "unknown"),
                "date":   date,
            })
            count += 1

        print(f"  ✓ {key}: {count} docs")

    except Exception as e:
        print(f"  ✗ {key}: {e}")

print(f"\nTotal docs collected: {len(docs)}")

if len(docs) == 0:
    print("No docs fetched — check your internet connection")
    sys.exit(1)

# Embed
print("\nEmbedding documents...")
from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts    = [d["text"] for d in docs]
emb      = np.array(embedder.encode(texts, show_progress_bar=True), dtype=np.float32)
faiss.normalize_L2(emb)

# Build index
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

# Save
idx_dir = Path("./data/faiss_index")
idx_dir.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(idx_dir / "index.faiss"))
(idx_dir / "documents.json").write_text(
    json.dumps(docs, indent=2), encoding="utf-8"
)

print(f"\n✓ FAISS index rebuilt: {len(docs)} documents")
print(f"✓ Saved to ./data/faiss_index/")

# Show breakdown
from collections import Counter
sources = Counter(d["source"] for d in docs)
print("\nBreakdown by source:")
for source, count in sorted(sources.items()):
    print(f"  {source}: {count}")