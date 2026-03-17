import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import faiss
import feedparser
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.core.llm import get_llm

settings = get_settings()

# ── Source trust weights ──────────────────────────────────────
SOURCE_WEIGHTS = {
    "pib.gov.in":         {"weight": 0.95, "tier": "government"},
    "imd.gov.in":         {"weight": 0.95, "tier": "government"},
    "ndma.gov.in":        {"weight": 0.95, "tier": "government"},
    "who.int":            {"weight": 0.90, "tier": "government"},
    "mohfw.gov.in":       {"weight": 0.90, "tier": "government"},
    "altnews.in":         {"weight": 0.95, "tier": "fact_check"},
    "boomlive.in":        {"weight": 0.95, "tier": "fact_check"},
    "vishvasnews.com":    {"weight": 0.90, "tier": "fact_check"},
    "thehindu.com":       {"weight": 0.90, "tier": "tier1_news"},
    "indianexpress.com":  {"weight": 0.90, "tier": "tier1_news"},
    "ndtv.com":           {"weight": 0.88, "tier": "tier1_news"},
    "ptinews.com":        {"weight": 0.90, "tier": "tier1_news"},
    "timesofindia.com":   {"weight": 0.80, "tier": "tier2_news"},
    "hindustantimes.com": {"weight": 0.80, "tier": "tier2_news"},
    "thewire.in":         {"weight": 0.82, "tier": "tier2_news"},
    "wikipedia.org":      {"weight": 0.85, "tier": "encyclopedia"},
}

# ── Claim type classification ─────────────────────────────────
TIME_ANCHORS = [
    "today", "tonight", "tomorrow", "yesterday",
    "right now", "currently", "this morning", "in the next",
    "aaj", "abhi", "kal", "abhi abhi",
]

VERIFY_PROMPT = PromptTemplate(
    input_variables=["claim", "sources"],
    template="""You are a crisis fact-checker for India's disaster management system.
Your job is to verify claims using ONLY the provided source documents.
Do NOT use any knowledge from your training data.

CLAIM: "{claim}"

TRUSTED SOURCE DOCUMENTS:
{sources}

INSTRUCTIONS:
- Base your verdict ONLY on the documents above
- Government sources (GOV) take priority over news sources
- Fact-check sources (FC) are strong evidence of false claims
- If documents contradict the claim → FALSE
- If documents support the claim → TRUE
- If documents are insufficient → UNVERIFIABLE

Credibility score 0-100:
  0-34  = FALSE   (contradicts sources or no evidence)
  35-69 = UNVERIFIABLE (insufficient evidence)
  70-100 = TRUE   (supported by trusted sources)

Respond ONLY with valid JSON, nothing else:
{{
  "verdict": "TRUE" or "FALSE" or "UNVERIFIABLE",
  "credibility_score": <integer 0-100>,
  "reasoning": "<2-3 sentence explanation citing sources>",
  "sources_used": ["<source name or URL>"]
}}"""
)

CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["claim"],
    template="""Classify this claim into exactly ONE category:

EPHEMERAL   - time-sensitive, changes within hours or days
             (weather, crisis alerts, rescue status, live events)
PERMANENT   - historical fact, does not change over time
             (deaths, laws passed, court verdicts, historical events)
SEMI_PERMANENT - changes slowly over months or years
             (policies, appointments, disease outbreaks)

Claim: "{claim}"

Respond with only one word: EPHEMERAL or PERMANENT or SEMI_PERMANENT"""
)


def freshness_weight(date_str: str, claim_type: str) -> float:
    try:
        from email.utils import parsedate_to_datetime
        published = parsedate_to_datetime(date_str)
        age = datetime.now(published.tzinfo) - published

        if claim_type == "PERMANENT":
            return 1.0

        if claim_type == "SEMI_PERMANENT":
            if age < timedelta(days=30):   return 1.0
            if age < timedelta(days=90):   return 0.9
            if age < timedelta(days=365):  return 0.7
            return 0.5

        # EPHEMERAL
        if age < timedelta(hours=2):   return 1.0
        if age < timedelta(hours=6):   return 0.9
        if age < timedelta(hours=24):  return 0.7
        if age < timedelta(days=3):    return 0.3
        return 0.1

    except Exception:
        return 0.5


class VerifyAgent:

    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self._index:    Optional[faiss.Index] = None
        self._documents: list = []
        self._chain     = None
        self._classify_chain = None
        self._ready     = False

    async def initialize(self):
        logger.info("Initialising VERIFY agent...")
        self._embedder = SentenceTransformer(settings.embedding_model)
        llm = get_llm()
        self._chain          = VERIFY_PROMPT    | llm | StrOutputParser()
        self._classify_chain = CLASSIFY_PROMPT  | llm | StrOutputParser()
        await self._load_or_build_index()
        self._ready = True
        logger.success("VERIFY agent ready ✓")

    async def _load_or_build_index(self):
        index_file = Path(settings.faiss_index_path) / "index.faiss"
        docs_file  = Path(settings.faiss_index_path) / "documents.json"

        if index_file.exists() and docs_file.exists():
            logger.info("Loading FAISS index from disk...")
            self._index     = faiss.read_index(str(index_file))
            self._documents = json.loads(docs_file.read_text(encoding="utf-8"))
            logger.info(f"Loaded {len(self._documents)} documents ✓")
        else:
            logger.info("No index found — building from RSS feeds...")
            await self._build_index()

    async def _build_index(self):
        sources = [
            (settings.pib_rss,    "pib.gov.in"),
            (settings.imd_rss,    "imd.gov.in"),
            (settings.who_rss,    "who.int"),
            (settings.ndma_rss,   "ndma.gov.in"),
            (settings.hindu_rss,  "thehindu.com"),
            (settings.ie_rss,     "indianexpress.com"),
            (settings.ndtv_rss,   "ndtv.com"),
            (settings.altnews_rss,"altnews.in"),
            (settings.boom_rss,   "boomlive.in"),
        ]

        rumour_phrases = [
            "viral message claims", "whatsapp forward",
            "rumours suggest", "unverified reports",
            "social media claims", "it is being claimed",
        ]

        docs = []
        for url, key in sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    title   = entry.get("title", "").strip()
                    summary = entry.get("summary", "").strip()
                    link    = entry.get("link", "")
                    date    = entry.get("published", "")

                    if not title:
                        continue

                    # Strip HTML
                    from bs4 import BeautifulSoup
                    summary = BeautifulSoup(
                        summary, "html.parser"
                    ).get_text(separator=" ").strip()[:500]

                    text = f"{title}. {summary}".strip()

                    # Skip rumour-reporting chunks
                    if any(p in text.lower() for p in rumour_phrases):
                        continue

                    docs.append({
                        "text":   text,
                        "source": key,
                        "url":    link,
                        "weight": SOURCE_WEIGHTS.get(
                            key, {"weight": 0.6}
                        )["weight"],
                        "tier":   SOURCE_WEIGHTS.get(
                            key, {"tier": "unknown"}
                        )["tier"],
                        "date":   date,
                    })

                logger.info(f"  {key}: {len(feed.entries)} entries")
            except Exception as e:
                logger.warning(f"  {key} failed: {e}")

        if not docs:
            logger.warning("No docs fetched — FAISS will be empty")
            return

        texts = [d["text"] for d in docs]
        logger.info(f"Embedding {len(texts)} documents...")
        emb = np.array(
            self._embedder.encode(texts, show_progress_bar=True),
            dtype=np.float32
        )
        faiss.normalize_L2(emb)

        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        idx_dir = Path(settings.faiss_index_path)
        idx_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(idx_dir / "index.faiss"))
        (idx_dir / "documents.json").write_text(
            json.dumps(docs, indent=2), encoding="utf-8"
        )

        self._index     = index
        self._documents = docs
        logger.success(f"FAISS index built: {len(docs)} docs ✓")

    async def rebuild_index(self):
        logger.info("Rebuilding FAISS index...")
        await self._build_index()

    def _classify_claim_type(self, claim: str) -> str:
        # Fast check for time anchors first
        claim_lower = claim.lower()
        if any(anchor in claim_lower for anchor in TIME_ANCHORS):
            return "EPHEMERAL"
        return "EPHEMERAL"  # default — LLM classification in next version

    def _retrieve(self, claim: str, claim_type: str, top_k: int = 5) -> list:
        if not self._index or self._index.ntotal == 0:
            return []

        emb = np.array(
            self._embedder.encode([claim]),
            dtype=np.float32
        )
        faiss.normalize_L2(emb)
        scores, indices = self._index.search(emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = {**self._documents[idx], "similarity": float(score)}

            # Apply freshness weight
            fresh = freshness_weight(doc.get("date", ""), claim_type)
            doc["final_weight"] = doc["weight"] * fresh
            results.append(doc)

        # Sort by final_weight
        results.sort(key=lambda x: x["final_weight"], reverse=True)
        return results

    def _format_sources(self, docs: list) -> str:
        lines = []
        for i, doc in enumerate(docs, 1):
            tier = doc.get("tier", "unknown").upper()
            tag  = "GOV" if tier == "GOVERNMENT" else \
                   "FC"  if tier == "FACT_CHECK"  else "NEWS"
            lines.append(
                f"[{i}] ({doc['source']}) [{tag}] "
                f"weight:{doc['final_weight']:.2f}\n{doc['text'][:300]}"
            )
        return "\n\n".join(lines)

    def _parse_llm_output(self, raw: str, retrieved: list) -> dict:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                return {
                    "verdict":           d.get("verdict", "UNVERIFIABLE"),
                    "credibility_score": int(d.get("credibility_score", 50)),
                    "reasoning":         d.get("reasoning", ""),
                    "sources":           d.get("sources_used", []),
                }
            except Exception:
                pass

        # Text fallback
        rl = raw.lower()
        if "false" in rl or "misinformation" in rl:
            verdict, score = "FALSE", 15
        elif "true" in rl or "confirmed" in rl:
            verdict, score = "TRUE", 82
        else:
            verdict, score = "UNVERIFIABLE", 50

        return {
            "verdict":           verdict,
            "credibility_score": score,
            "reasoning":         raw[:300],
            "sources":           [d["url"] for d in retrieved[:2]],
        }

    async def verify(self, claim: str, platform: str = "manual") -> dict:
        if not self._ready:
            await self.initialize()

        logger.info(f"Verifying: {claim[:80]}...")

        # Step 1 — classify claim type
        claim_type = self._classify_claim_type(claim)

        # Step 2 — check template cache first (fast path)
        cached = await self._check_template_cache(claim)
        if cached:
            logger.info("Cache hit — returning cached verdict")
            return {**cached, "from_cache": True,
                    "verified_at": datetime.utcnow().isoformat()}

        # Step 3 — retrieve from FAISS
        retrieved = self._retrieve(claim, claim_type)

        if not retrieved:
            return self._no_sources_result(claim, claim_type)

        # Step 4 — check minimum relevance
        if retrieved[0]["similarity"] < 0.3:
            return self._no_sources_result(claim, claim_type)

        # Step 5 — call Groq
        source_ctx = self._format_sources(retrieved)
        try:
            raw    = await self._chain.ainvoke({
                "claim":   claim,
                "sources": source_ctx,
            })
            result = self._parse_llm_output(raw, retrieved)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._error_result(claim, str(e))

        # Step 6 — trust weight adjustment
        top_w = max(
            (d.get("final_weight", 0.6) for d in retrieved[:3]),
            default=0.6
        )
        score = result["credibility_score"]
        if result["verdict"] == "TRUE" and top_w >= 0.9:
            score = min(100, int(score * 1.1))
        elif result["verdict"] == "FALSE" and top_w < 0.7:
            score = max(0, int(score * 0.9))
        result["credibility_score"] = score

        final = {
            **result,
            "claim":          claim,
            "claim_type":     claim_type,
            "platform":       platform,
            "retrieved_docs": retrieved,
            "from_cache":     False,
            "verified_at":    datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Verdict: {final['verdict']} | "
            f"Score: {final['credibility_score']} | "
            f"{claim[:50]}..."
        )
        return final

    async def _check_template_cache(self, claim: str) -> Optional[dict]:
        try:
            from backend.db.mongodb import get_db
            db  = get_db()
            emb = self._embedder.encode([claim])[0].tolist()

            templates = await db.templates.find(
                {}, {"template_text": 1, "verdict": 1,
                     "credibility_score": 1, "explanation": 1,
                     "source": 1, "embedding": 1}
            ).to_list(length=200)

            for t in templates:
                if not t.get("embedding"):
                    continue
                a = np.array(emb)
                b = np.array(t["embedding"])
                sim = float(
                    np.dot(a, b) /
                    (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                )
                if sim > 0.85:
                    # Update occurrence count
                    await db.templates.update_one(
                        {"_id": t["_id"]},
                        {"$inc": {"occurrence_count": 1},
                         "$set": {"last_seen": datetime.utcnow()}}
                    )
                    return {
                        "verdict":           t["verdict"],
                        "credibility_score": t["credibility_score"],
                        "reasoning":         t.get("explanation", ""),
                        "sources":           [t.get("source", "")],
                        "claim":             claim,
                    }
        except Exception as e:
            logger.warning(f"Template cache check failed: {e}")
        return None

    def _no_sources_result(self, claim: str, claim_type: str) -> dict:
        return {
            "claim":             claim,
            "verdict":           "UNVERIFIABLE",
            "credibility_score": 40,
            "reasoning":         "No relevant trusted sources found. "
                                 "Check ndma.gov.in or mausam.imd.gov.in "
                                 "for official updates.",
            "sources":           [],
            "claim_type":        claim_type,
            "retrieved_docs":    [],
            "from_cache":        False,
            "verified_at":       datetime.utcnow().isoformat(),
        }

    def _error_result(self, claim: str, error: str) -> dict:
        return {
            "claim":             claim,
            "verdict":           "UNVERIFIABLE",
            "credibility_score": 0,
            "reasoning":         f"Verification error: {error}",
            "sources":           [],
            "claim_type":        "EPHEMERAL",
            "retrieved_docs":    [],
            "from_cache":        False,
            "verified_at":       datetime.utcnow().isoformat(),
        }


# ── Singleton ─────────────────────────────────────────────────
_agent: Optional[VerifyAgent] = None


async def get_verify_agent() -> VerifyAgent:
    global _agent
    if _agent is None:
        _agent = VerifyAgent()
        await _agent.initialize()
    return _agent