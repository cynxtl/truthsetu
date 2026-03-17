"""
TruthSetu — Government Source Fetcher
Workarounds for broken Indian government RSS feeds
"""
import requests
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}

docs = []

# ── PIB — scrape directly from website ────────────────────────
print("Fetching PIB...")
try:
    urls = [
        "https://pib.gov.in/allRel.aspx",
        "https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1",
    ]
    r = requests.get(
        "https://pib.gov.in/allRel.aspx",
        headers=HEADERS, timeout=15
    )
    soup = BeautifulSoup(r.content, "html.parser")

    # Find press release links and titles
    items = soup.find_all("a", href=re.compile(r"PressRelease"))
    count = 0
    for item in items[:30]:
        title = item.get_text(strip=True)
        link  = "https://pib.gov.in/" + item.get("href", "")
        if len(title) > 20:
            docs.append({
                "text":   f"PIB Press Release: {title}",
                "source": "pib.gov.in",
                "url":    link,
                "weight": 0.95,
                "tier":   "government",
                "date":   datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"),
            })
            count += 1
    print(f"  ✓ PIB: {count} press releases")
except Exception as e:
    print(f"  ✗ PIB failed: {e}")

# ── IMD — use alternative endpoints ───────────────────────────
print("Fetching IMD...")
IMD_URLS = [
    "https://mausam.imd.gov.in/responsive/cycloneWarning.php",
    "https://mausam.imd.gov.in/responsive/all_india_forcast_bulletin.php",
    "https://mausam.imd.gov.in/responsive/rainfallinformation_state.php",
]
try:
    for url in IMD_URLS:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)[:1000]
        if len(text) > 100:
            docs.append({
                "text":   f"IMD Advisory: {text}",
                "source": "imd.gov.in",
                "url":    url,
                "weight": 0.95,
                "tier":   "government",
                "date":   datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"),
            })
    print(f"  ✓ IMD: {len([d for d in docs if d['source']=='imd.gov.in'])} pages")
except Exception as e:
    print(f"  ✗ IMD failed: {e}")

# ── NDMA — scrape press releases ──────────────────────────────
print("Fetching NDMA...")
try:
    r = requests.get(
        "https://ndma.gov.in/Media/Press-Releases",
        headers=HEADERS, timeout=15
    )
    soup = BeautifulSoup(r.content, "html.parser")
    items = soup.find_all(["h3", "h4", "a"], limit=40)
    count = 0
    for item in items:
        text = item.get_text(strip=True)
        if len(text) > 30:
            docs.append({
                "text":   f"NDMA Advisory: {text}",
                "source": "ndma.gov.in",
                "url":    "https://ndma.gov.in/Media/Press-Releases",
                "weight": 0.95,
                "tier":   "government",
                "date":   datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"),
            })
            count += 1
    print(f"  ✓ NDMA: {count} items")
except Exception as e:
    print(f"  ✗ NDMA failed: {e}")

# ── WHO — use working RSS endpoint ────────────────────────────
print("Fetching WHO...")
WHO_URLS = [
    "https://www.who.int/feeds/entity/csr/don/en/rss.xml",
    "https://www.who.int/feeds/entity/mediacentre/news/en/rss.xml",
    "https://extranet.who.int/inf-fs/en/index.html",
]
try:
    import feedparser
    fetched = 0
    for url in WHO_URLS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:20]:
            title   = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            if title:
                text = BeautifulSoup(
                    f"{title}. {summary}", "html.parser"
                ).get_text()[:500]
                docs.append({
                    "text":   text,
                    "source": "who.int",
                    "url":    entry.get("link", url),
                    "weight": 0.90,
                    "tier":   "government",
                    "date":   entry.get("published", ""),
                })
                fetched += 1
        if fetched > 0:
            break
    print(f"  ✓ WHO: {fetched} entries")
except Exception as e:
    print(f"  ✗ WHO failed: {e}")

# ── Static trusted facts ───────────────────────────────────────
# Since government sites are unreliable, add authoritative
# static facts that never change — VERIFY can always use these
print("Adding static trusted facts...")

STATIC_FACTS = [
    "NDMA advises citizens to avoid low-lying areas, basements and "
    "flooded roads during flood emergencies. Move to higher ground immediately.",

    "IMD issues weather warnings including red alert (take action), "
    "orange alert (be prepared) and yellow alert (be aware) for severe weather.",

    "During cyclone warnings, NDMA recommends staying indoors, "
    "avoiding coastal areas, and following official evacuation orders only.",

    "PIB Fact Check is the official Government of India portal for "
    "debunking misinformation about government schemes and policies.",

    "The India Meteorological Department (IMD) is the only official "
    "source for weather forecasts and cyclone track predictions in India.",

    "NDMA is India's National Disaster Management Authority responsible "
    "for coordinating disaster response and issuing official alerts.",

    "WHO is the United Nations health agency. It declares Public Health "
    "Emergencies of International Concern (PHEIC) for serious disease outbreaks.",

    "Election Commission of India (ECI) is the sole authority for "
    "announcing election schedules, results and code of conduct.",

    "Reserve Bank of India (RBI) is the only authority that can "
    "demonetize currency or announce changes to legal tender.",

    "EVM (Electronic Voting Machines) used in Indian elections are "
    "standalone devices not connected to internet or any network.",

    "Earthquake prediction is not scientifically possible. "
    "No agency can predict exact time and location of earthquakes.",

    "The Supreme Court of India struck down the Aadhaar-bank account "
    "mandatory linking requirement in 2018.",

    "India has a multi-tier flood warning system: "
    "CWC monitors river levels and issues flood forecasts for major rivers.",

    "During pandemics, MoHFW (Ministry of Health and Family Welfare) "
    "is the official source for health advisories in India.",

    "AERB (Atomic Energy Regulatory Board) monitors all nuclear "
    "facilities in India and publishes safety reports publicly.",
]

for i, fact in enumerate(STATIC_FACTS):
    docs.append({
        "text":   fact,
        "source": "truthsetu.static",
        "url":    "https://ndma.gov.in",
        "weight": 0.88,
        "tier":   "government",
        "date":   datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"),
    })

print(f"  ✓ Added {len(STATIC_FACTS)} static trusted facts")

# ── Save and merge with existing docs ─────────────────────────
print(f"\nTotal new gov docs: {len(docs)}")

# Load existing docs
existing_file = Path("./data/faiss_index/documents.json")
existing = []
if existing_file.exists():
    existing = json.loads(existing_file.read_text(encoding="utf-8"))
    # Remove old gov docs to replace with fresh ones
    existing = [
        d for d in existing
        if d["source"] not in [
            "pib.gov.in", "imd.gov.in", "ndma.gov.in",
            "who.int", "truthsetu.static"
        ]
    ]
    print(f"Kept {len(existing)} existing news docs")

all_docs = existing + docs
print(f"Total combined: {len(all_docs)} docs")

# Rebuild FAISS with all docs
print("\nRebuilding FAISS index...")
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts    = [d["text"] for d in all_docs]
emb      = np.array(
    embedder.encode(texts, show_progress_bar=True, batch_size=32),
    dtype=np.float32
)
faiss.normalize_L2(emb)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

idx_dir = Path("./data/faiss_index")
faiss.write_index(index, str(idx_dir / "index.faiss"))
existing_file.write_text(
    json.dumps(all_docs, indent=2), encoding="utf-8"
)

print(f"\n✓ FAISS rebuilt: {len(all_docs)} total documents")
print("\nBreakdown:")
from collections import Counter
for source, count in sorted(Counter(
    d["source"] for d in all_docs
).items()):
    print(f"  {source}: {count}")