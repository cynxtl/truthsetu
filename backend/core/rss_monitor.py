"""
TruthSetu — RSS Monitor
========================
Three jobs:
  1. Auto-refresh FAISS index every 30 minutes
  2. Monitor Alt News + BOOM + Vishvas → seed LEARN templates
  3. Fast-path RSS search for VERIFY (before web search)

Full article fetching strategy:
  Fact-checkers (Alt News, BOOM, Vishvas) → fetch full text
  News sources (Hindu, NDTV etc.)         → RSS summary only
  Government sources (PIB, IMD, NDMA)     → RSS summary only
  Fallback: DDG search snippets via VERIFY web search tool
"""

import json
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from loguru import logger
from bs4 import BeautifulSoup
import feedparser
import numpy as np
import httpx

from backend.core.config import get_settings

settings = get_settings()

# ── All RSS sources ───────────────────────────────────────────
RSS_SOURCES = [
    # Government — highest trust, RSS summary only
    {
        "url":     "https://pib.gov.in/RssMain.aspx?ModID=6&Lang=1",
        "key":     "pib.gov.in",
        "weight":  0.95,
        "tier":    "government",
        "refresh": 30,
        "fetch_full": False,
    },
    {
        "url":     "https://mausam.imd.gov.in/imd_latest/contents/rss-feed.php",
        "key":     "imd.gov.in",
        "weight":  0.95,
        "tier":    "government",
        "refresh": 15,
        "fetch_full": False,
    },
    {
        "url":     "https://ndma.gov.in/RSS/ndma.xml",
        "key":     "ndma.gov.in",
        "weight":  0.95,
        "tier":    "government",
        "refresh": 15,
        "fetch_full": False,
    },
    {
        "url":     "https://www.who.int/rss-feeds/news-releases.xml",
        "key":     "who.int",
        "weight":  0.90,
        "tier":    "government",
        "refresh": 60,
        "fetch_full": False,
    },
    # Fact-checkers — fetch full article text
    {
        "url":     "https://www.altnews.in/feed/",
        "key":     "altnews.in",
        "weight":  0.95,
        "tier":    "fact_check",
        "refresh": 15,
        "fetch_full": True,   # full article for best accuracy
        "monitor":   True,    # also feed into LEARN
    },
    {
        "url":     "https://www.boomlive.in/feed",
        "key":     "boomlive.in",
        "weight":  0.95,
        "tier":    "fact_check",
        "refresh": 15,
        "fetch_full": True,
        "monitor":   True,
    },
    {
        "url":     "https://www.vishvasnews.com/feed/",
        "key":     "vishvasnews.com",
        "weight":  0.90,
        "tier":    "fact_check",
        "refresh": 30,
        "fetch_full": True,
        "monitor":   True,
    },
    # Tier 1 news — RSS summary only (DDG handles depth)
    {
        "url":     "https://www.thehindu.com/news/feeder/default.rss",
        "key":     "thehindu.com",
        "weight":  0.90,
        "tier":    "tier1_news",
        "refresh": 15,
        "fetch_full": False,
    },
    {
        "url":     "https://indianexpress.com/feed/",
        "key":     "indianexpress.com",
        "weight":  0.90,
        "tier":    "tier1_news",
        "refresh": 15,
        "fetch_full": False,
    },
    {
        "url":     "https://feeds.feedburner.com/ndtvnews-top-stories",
        "key":     "ndtv.com",
        "weight":  0.88,
        "tier":    "tier1_news",
        "refresh": 15,
        "fetch_full": False,
    },
    # Tier 2 news — RSS summary only
    {
        "url":     "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "key":     "timesofindia.com",
        "weight":  0.80,
        "tier":    "tier2_news",
        "refresh": 30,
        "fetch_full": False,
    },
    {
        "url":     "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
        "key":     "hindustantimes.com",
        "weight":  0.80,
        "tier":    "tier2_news",
        "refresh": 30,
        "fetch_full": False,
    },
]

RUMOUR_PHRASES = [
    "viral message claims", "whatsapp forward",
    "rumours suggest", "unverified reports",
    "social media claims", "it is being claimed",
    "fake news spreads", "misleading post",
]

CRISIS_KEYWORDS = [
    "cyclone", "flood", "earthquake", "tsunami", "pandemic",
    "virus", "outbreak", "epidemic", "evacuation", "disaster",
    "emergency", "alert", "warning", "rescue", "trapped",
    "vaccine", "covid", "corona", "lockdown", "quarantine",
    "election", "evm", "dam", "radiation", "explosion",
    "riot", "violence", "curfew", "contaminated", "missing",
]

# Global state
_last_fetched:   dict[str, datetime] = {}
_recent_entries: list[dict]          = []
_fetched_urls:   set[str]            = set()  # avoid duplicate fetches


class RSSMonitor:

    def __init__(self):
        self._embedder  = None
        self._http      = None
        self._ready     = False

    async def initialize(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Initialising RSS Monitor...")
        self._embedder = SentenceTransformer(settings.embedding_model)
        self._ready    = True
        logger.success("RSS Monitor ready ✓")

    # ─────────────────────────────────────────────────────────
    # Job 1 — Auto-refresh FAISS index
    # ─────────────────────────────────────────────────────────

    async def refresh_faiss_index(self):
        """
        Fetch all RSS sources and update FAISS with new entries.
        Called every 30 minutes by scheduler.
        """
        if not self._ready:
            await self.initialize()

        logger.info("RSS auto-refresh starting...")
        now      = datetime.now(timezone.utc)
        new_docs = []

        for source in RSS_SOURCES:
            # Check if this source needs refreshing
            last = _last_fetched.get(source["key"])
            if last:
                age_mins = (now - last).total_seconds() / 60
                if age_mins < source["refresh"]:
                    continue

            try:
                docs = await self._fetch_source(source)
                new_docs.extend(docs)
                _last_fetched[source["key"]] = now
                if docs:
                    logger.info(
                        f"  {source['key']}: {len(docs)} entries "
                        f"({'full text' if source['fetch_full'] else 'summary'})"
                    )
            except Exception as e:
                logger.warning(f"  {source['key']} failed: {e}")

        if not new_docs:
            logger.info("RSS refresh: no new entries this cycle")
            return

        # Update recent entries cache (last 6 hours)
        global _recent_entries
        cutoff          = now - timedelta(hours=6)
        _recent_entries = [
            e for e in _recent_entries
            if self._parse_date(e.get("date","")) > cutoff
        ]
        _recent_entries = (new_docs + _recent_entries)[:500]

        # Update FAISS with new docs
        await self._update_faiss(new_docs)

        logger.success(
            f"RSS refresh complete: {len(new_docs)} new docs added"
        )

    # ─────────────────────────────────────────────────────────
    # Fetch a single RSS source
    # ─────────────────────────────────────────────────────────

    async def _fetch_source(self, source: dict) -> list[dict]:
        """
        Fetch one RSS source.
        Full article fetched for fact-checkers.
        RSS summary used for news + government sources.
        """
        feed = feedparser.parse(source["url"])
        docs = []

        # For fact-checkers, fetch articles concurrently
        if source.get("fetch_full"):
            tasks = []
            entries_to_process = []

            for entry in feed.entries[:20]:
                title = entry.get("title", "").strip()
                link  = entry.get("link",  "")
                if not title or not link:
                    continue
                if link in _fetched_urls:
                    continue
                entries_to_process.append(entry)
                tasks.append(self._fetch_full_article(link))

            # Fetch all articles concurrently
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for entry, full_text in zip(entries_to_process, results):
                    title   = entry.get("title", "").strip()
                    summary = entry.get("summary", "").strip()
                    link    = entry.get("link",  "")
                    date    = entry.get("published", "")

                    # Clean RSS summary as fallback
                    summary_clean = BeautifulSoup(
                        summary, "html.parser"
                    ).get_text(separator=" ").strip()[:500]

                    # Use full text if successfully fetched
                    if isinstance(full_text, str) and len(full_text) > 200:
                        text = f"{title}.\n\n{full_text}"
                        content_type = "full_article"
                        _fetched_urls.add(link)
                    else:
                        text = f"{title}. {summary_clean}"
                        content_type = "rss_summary"

                    if any(p in text.lower() for p in RUMOUR_PHRASES):
                        continue

                    docs.append({
                        "text":         text[:3000],
                        "source":       source["key"],
                        "url":          link,
                        "weight":       source["weight"],
                        "tier":         source["tier"],
                        "date":         date,
                        "content_type": content_type,
                        "monitor":      source.get("monitor", False),
                    })

        else:
            # News + government sources — use RSS summary
            for entry in feed.entries[:30]:
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                link    = entry.get("link",  "")
                date    = entry.get("published", "")

                if not title:
                    continue

                summary_clean = BeautifulSoup(
                    summary, "html.parser"
                ).get_text(separator=" ").strip()[:500]

                text = f"{title}. {summary_clean}".strip()

                if any(p in text.lower() for p in RUMOUR_PHRASES):
                    continue

                docs.append({
                    "text":         text,
                    "source":       source["key"],
                    "url":          link,
                    "weight":       source["weight"],
                    "tier":         source["tier"],
                    "date":         date,
                    "content_type": "rss_summary",
                    "monitor":      False,
                })

        return docs

    # ─────────────────────────────────────────────────────────
    # Full article fetcher (fact-checkers only)
    # ─────────────────────────────────────────────────────────

    async def _fetch_full_article(self, url: str) -> Optional[str]:
        """
        Fetch full article text using trafilatura.
        Only called for fact-check sources.
        Returns None on failure — caller falls back to RSS summary.
        """
        try:
            import trafilatura

            async with httpx.AsyncClient(
                timeout=15,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36",
                    "Accept":          "text/html,application/xhtml+xml",
                    "Accept-Language": "en-IN,en;q=0.9",
                }
            ) as client:
                response = await client.get(url)

            if response.status_code != 200:
                return None

            # Extract clean text with trafilatura
            text = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=True,
            )

            if text and len(text) > 200:
                # Cap at 2500 chars — enough for FAISS, not too heavy
                return text[:2500].strip()

            return None

        except Exception as e:
            logger.debug(f"Full fetch failed for {url}: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # FAISS index update
    # ─────────────────────────────────────────────────────────

    async def _update_faiss(self, new_docs: list[dict]):
        """Add new documents to existing FAISS index incrementally."""
        try:
            import faiss

            index_file = Path(settings.faiss_index_path) / "index.faiss"
            docs_file  = Path(settings.faiss_index_path) / "documents.json"

            if not index_file.exists():
                logger.warning("FAISS index not found — skipping update")
                return

            # Load existing
            index    = faiss.read_index(str(index_file))
            existing = json.loads(docs_file.read_text(encoding="utf-8"))

            # Deduplicate — don't add docs already in index
            existing_urls = {d.get("url", "") for d in existing}
            truly_new     = [
                d for d in new_docs
                if d.get("url", "") not in existing_urls
            ]

            if not truly_new:
                logger.info("FAISS update: all entries already indexed")
                return

            # Embed new docs
            texts = [d["text"] for d in truly_new]
            emb   = np.array(
                self._embedder.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=32,
                ),
                dtype=np.float32
            )
            faiss.normalize_L2(emb)

            # Add to index
            index.add(emb)

            # Merge old entries
            all_docs = existing + truly_new

            # Save
            faiss.write_index(index, str(index_file))
            docs_file.write_text(
                json.dumps(all_docs, indent=2),
                encoding="utf-8"
            )

            logger.info(
                f"FAISS updated: +{len(truly_new)} new docs | "
                f"{index.ntotal} total vectors | "
                f"{len(all_docs)} total docs"
            )

        except Exception as e:
            logger.error(f"FAISS update failed: {e}")

    # ─────────────────────────────────────────────────────────
    # Job 2 — Monitor fact-checkers → seed LEARN
    # ─────────────────────────────────────────────────────────

    async def monitor_fact_checkers(self):
        """
        Watch Alt News, BOOM, Vishvas for new fact-checks.
        Extract debunked claims → store in MongoDB templates.
        Called every 15 minutes by scheduler.
        """
        if not self._ready:
            await self.initialize()

        logger.info("Monitoring fact-checker feeds...")

        monitor_sources = [s for s in RSS_SOURCES if s.get("monitor")]

        for source in monitor_sources:
            try:
                docs = await self._fetch_source(source)
                tasks = [
                    self._process_fact_check(doc, source["key"])
                    for doc in docs
                    if doc.get("monitor")
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                logger.warning(f"Monitor {source['key']} failed: {e}")

    async def _process_fact_check(self, doc: dict, source: str):
        """
        Extract false claim from fact-check article.
        Store in MongoDB templates for instant VERIFY cache hits.
        """
        try:
            from backend.db.mongodb import get_db
            from backend.core.llm import get_llm
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            db = get_db()

            # Skip if already processed
            existing = await db.templates.find_one({"source_url": doc["url"]})
            if existing:
                return

            # Use first 500 chars (headline + start of article)
            content = doc["text"][:500]

            # Extract the false claim being debunked
            prompt = PromptTemplate(
                input_variables=["content", "source"],
                template="""This is from {source}, an Indian fact-checking website.
Extract the FALSE claim being debunked.

Article content:
"{content}"

Rules:
- Extract only the false claim (what people are incorrectly sharing)
- Write it as a single clear sentence
- If no clear false claim found, respond: NO_CLAIM

Respond with the false claim only or NO_CLAIM:"""
            )

            chain  = prompt | get_llm() | StrOutputParser()
            result = await chain.ainvoke({
                "content": content,
                "source":  source,
            })
            result = result.strip()

            if result == "NO_CLAIM" or len(result) < 10:
                return

            # Detect crisis type from content
            crisis_type = self._detect_crisis_type(content)

            # Generate embedding for the extracted claim
            emb = self._embedder.encode([result])[0].tolist()

            # Store in templates collection
            await db.templates.update_one(
                {"template_text": result},
                {"$setOnInsert": {
                    "template_text":     result,
                    "example_claim":     result,
                    "crisis_type":       crisis_type,
                    "claim_type":        "EPHEMERAL",
                    "verdict":           "FALSE",
                    "credibility_score": 5,
                    "source":            source,
                    "source_url":        doc["url"],
                    "explanation":       (
                        f"Debunked by {source}. "
                        f"{doc['text'][:300]}"
                    ),
                    "embedding":         emb,
                    "first_seen":        datetime.now(timezone.utc),
                    "last_seen":         datetime.now(timezone.utc),
                    "occurrence_count":  1,
                    "auto_seeded":       True,
                }},
                upsert=True
            )

            logger.info(
                f"Auto-seeded template [{crisis_type}] "
                f"from {source}: {result[:70]}"
            )

        except Exception as e:
            logger.debug(f"Fact-check processing failed: {e}")

    def _detect_crisis_type(self, text: str) -> str:
        """Detect crisis type from article text using keywords."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["cyclone", "hurricane", "typhoon", "storm"]):
            return "cyclone"
        if any(w in text_lower for w in ["flood", "rain", "dam", "river"]):
            return "flood"
        if any(w in text_lower for w in ["earthquake", "tremor", "seismic"]):
            return "earthquake"
        if any(w in text_lower for w in ["covid", "corona", "pandemic", "virus",
                                          "vaccine", "outbreak", "epidemic"]):
            return "pandemic"
        if any(w in text_lower for w in ["election", "evm", "vote", "ballot",
                                          "polling", "candidate"]):
            return "election"
        if any(w in text_lower for w in ["riot", "communal", "violence",
                                          "attack", "mob"]):
            return "communal"
        if any(w in text_lower for w in ["government", "policy", "scheme",
                                          "law", "ban", "rbi", "rupee"]):
            return "policy"
        return "general"

    # ─────────────────────────────────────────────────────────
    # Job 3 — Fast-path RSS search for VERIFY
    # ─────────────────────────────────────────────────────────

    async def search_recent_rss(
        self, query: str, max_results: int = 5
    ) -> list[dict]:
        """
        Search recent RSS entries (last 6 hours) for relevant content.
        Used by VERIFY as fast middle path before web search.
        Faster than DuckDuckGo, no rate limits.
        """
        if not self._ready:
            await self.initialize()

        if not _recent_entries:
            logger.info("RSS cache empty — triggering refresh")
            await self.refresh_faiss_index()

        if not _recent_entries:
            return []

        # Embed query
        import faiss
        query_emb = np.array(
            self._embedder.encode([query]),
            dtype=np.float32
        )
        faiss.normalize_L2(query_emb)

        # Score all recent entries
        results = []
        for entry in _recent_entries:
            try:
                entry_emb = np.array(
                    self._embedder.encode([entry["text"][:500]]),
                    dtype=np.float32
                )
                faiss.normalize_L2(entry_emb)
                sim = float(np.dot(query_emb[0], entry_emb[0]))

                if sim > 0.35:
                    results.append({
                        **entry,
                        "similarity":    sim,
                        "final_weight":  entry["weight"],
                    })
            except Exception:
                continue

        results.sort(key=lambda x: x["similarity"], reverse=True)

        if results:
            logger.info(
                f"RSS fast-path: {len(results)} matches | "
                f"top: {results[0]['source']} "
                f"(sim={results[0]['similarity']:.3f})"
            )

        return results[:max_results]

    # ─────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────

    def _parse_date(self, date_str: str) -> datetime:
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except Exception:
            return datetime.now(timezone.utc) - timedelta(days=1)


# ── Singleton ─────────────────────────────────────────────────
_monitor: Optional[RSSMonitor] = None


async def get_rss_monitor() -> RSSMonitor:
    global _monitor
    if _monitor is None:
        _monitor = RSSMonitor()
        await _monitor.initialize()
    return _monitor