"""
TruthSetu — VERIFY Agent (with web search)
==========================================
Pipeline:
  1. Check template cache (instant)
  2. FAISS search (local knowledge base)
  3. If evidence weak → Groq calls web search tool
  4. Groq reasons over FAISS + web results
  5. Return structured verdict
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
from groq import Groq
from loguru import logger
import feedparser
from bs4 import BeautifulSoup

from backend.core.config import get_settings

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
    "truthsetu.static":   {"weight": 0.88, "tier": "government"},
    "web_search":         {"weight": 0.70, "tier": "web"},
}

# ── Time anchors for claim classification ────────────────────
TIME_ANCHORS = [
    "today", "tonight", "tomorrow", "yesterday",
    "right now", "currently", "this morning", "in the next",
    "aaj", "abhi", "kal", "abhi abhi",
]

# ── Web search tool definition for Groq ──────────────────────
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for recent news, official statements, and "
            "verified information about a crisis claim in India. "
            "Use this when the provided documents are insufficient, "
            "outdated, or do not directly address the claim. "
            "Always search for Indian government sources like IMD, "
            "NDMA, PIB, WHO India when relevant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Specific search query. Include location, "
                        "crisis type, and time context. "
                        "Example: 'IMD cyclone Chennai warning today 2026'"
                    )
                }
            },
            "required": ["query"]
        }
    }
}

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are TruthSetu, India's crisis fact-checking AI assistant.

Your job is to help citizens understand what is actually known about a claim
based on current news and official sources.

CRITICAL RULES:
1. NEVER say an event is happening unless official sources explicitly confirm it NOW
2. "Tensions", "possibility", "risk", "fear of" — these do NOT mean the event is happening
3. Summarise what sources actually say — do not interpret beyond what is written
4. Be honest about uncertainty
5. Keep summary to 2-3 sentences maximum

VERDICT CRITERIA:
- FALSE:       Sources directly and clearly contradict the claim
- CONFIRMED:   Official sources explicitly confirm the event is happening RIGHT NOW
- UNVERIFIED:  Sources are related but do not directly confirm or deny the claim
- NO_INFO:     No relevant sources found at all

EXAMPLES OF CORRECT BEHAVIOUR:
  Claim: "War starting today in India"
  Sources: Articles about India-Pakistan tensions, border incidents
  WRONG verdict: CONFIRMED — war is starting
  RIGHT verdict: UNVERIFIED
  RIGHT summary: "Sources report heightened India-Pakistan border tensions and
                  military activity, but no official war declaration has been
                  made by either government."

  Claim: "Cyclone hitting Chennai in 2 hours"
  Sources: IMD bulletin saying cyclone is 500km away
  RIGHT verdict: FALSE
  RIGHT summary: "IMD confirms the cyclone is currently 500km from Chennai
                  coast with no immediate threat to the city."

  Claim: "Government announced free vaccine distribution"
  Sources: PIB press release confirming vaccine drive
  RIGHT verdict: CONFIRMED
  RIGHT summary: "PIB confirms the government has announced a free vaccination
                  drive starting Monday across all districts." """


# ── User prompt template ──────────────────────────────────────
def build_user_prompt(claim: str, docs_context: str) -> str:
    return f"""Verify this claim: "{claim}"

SOURCES:
{docs_context if docs_context else "No relevant documents found."}

Analyse the sources carefully. Return ONLY this JSON:
{{
  "verdict": "FALSE" or "CONFIRMED" or "UNVERIFIED" or "NO_INFO",
  "credibility_score": <integer 0-100>,
  "reasoning": "<2-3 sentence summary of what sources actually say — not what you think, what sources say>",
  "sources_used": ["<source names>"],
  "web_searched": true or false
}}

Remember:
- Only say CONFIRMED if a source explicitly states the event is happening NOW
- If sources show tensions/risks/possibilities → use UNVERIFIED
- If sources directly contradict the claim → use FALSE"""


def freshness_weight(date_str: str, claim_type: str) -> float:
    try:
        from email.utils import parsedate_to_datetime
        published = parsedate_to_datetime(date_str)
        age = datetime.now(published.tzinfo) - published

        if claim_type == "PERMANENT":
            return 1.0
        if claim_type == "SEMI_PERMANENT":
            if age < timedelta(days=30):  return 1.0
            if age < timedelta(days=90):  return 0.9
            if age < timedelta(days=365): return 0.7
            return 0.5
        # EPHEMERAL
        if age < timedelta(hours=2):  return 1.0
        if age < timedelta(hours=6):  return 0.9
        if age < timedelta(hours=24): return 0.7
        if age < timedelta(days=3):   return 0.3
        return 0.1
    except Exception:
        return 0.5


class VerifyAgent:

    def __init__(self):
        self._embedder:  Optional[SentenceTransformer] = None
        self._index:     Optional[faiss.Index] = None
        self._documents: list = []
        self._groq:      Optional[Groq] = None
        self._ready = False

    async def initialize(self):
        logger.info("Initialising VERIFY agent...")
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(settings.embedding_model)
        self._groq = Groq(api_key=settings.groq_api_key)
        await self._load_or_build_index()
        self._ready = True
        logger.success("VERIFY agent ready ✓")

    # ── Index management ──────────────────────────────────────

    async def _load_or_build_index(self):
        index_file = Path(settings.faiss_index_path) / "index.faiss"
        docs_file  = Path(settings.faiss_index_path) / "documents.json"

        if index_file.exists() and docs_file.exists():
            logger.info("Loading FAISS index from disk...")
            self._index     = faiss.read_index(str(index_file))
            self._documents = json.loads(
                docs_file.read_text(encoding="utf-8")
            )
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

                    summary = BeautifulSoup(
                        summary, "html.parser"
                    ).get_text(separator=" ").strip()[:500]

                    text = f"{title}. {summary}".strip()

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

    # ── Web search ────────────────────────────────────────────

    async def _web_search(self, query: str) -> list[dict]:
        """
        Search the web using DuckDuckGo.
        Returns list of {title, url, snippet} dicts.
        """
        try:
            from ddgs import DDGS

            logger.info(f"Web search: {query}")
            results = []

            with DDGS() as ddgs:
                for r in ddgs.text(
                    query,
                    region="in-en",        # India English results
                    safesearch="moderate",
                    max_results=5,
                ):
                    results.append({
                        "text":   f"{r.get('title','')}. {r.get('body','')}",
                        "source": "web_search",
                        "url":    r.get("href", ""),
                        "weight": 0.70,
                        "tier":   "web",
                        "date":   "",
                        "query":  query,
                    })

            logger.info(f"Web search returned {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    # ── FAISS retrieval ───────────────────────────────────────

    def _classify_claim_type(self, claim: str) -> str:
        claim_lower = claim.lower()
        if any(anchor in claim_lower for anchor in TIME_ANCHORS):
            return "EPHEMERAL"
        return "EPHEMERAL"

    def _retrieve(
        self, claim: str, claim_type: str, top_k: int = 5
    ) -> list:
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
            fresh = freshness_weight(doc.get("date", ""), claim_type)
            doc["final_weight"] = doc["weight"] * fresh
            results.append(doc)

        results.sort(key=lambda x: x["final_weight"], reverse=True)
        return results if results else []

    def _format_docs(self, docs: list) -> str:
        if not docs:
            return "No documents found."
        lines = []
        for i, doc in enumerate(docs, 1):
            tier = doc.get("tier", "unknown").upper()
            tag  = "GOV" if tier == "GOVERNMENT" else \
                   "FC"  if tier == "FACT_CHECK"  else \
                   "WEB" if tier == "WEB"          else "NEWS"
            lines.append(
                f"[{i}] [{tag}] {doc['source']}\n"
                f"{doc['text'][:300]}"
            )
        return "\n\n".join(lines)

    # ── Template cache ────────────────────────────────────────

    async def _check_template_cache(
        self, claim: str
    ) -> Optional[dict]:
        try:
            from backend.db.mongodb import get_db
            db  = get_db()
            emb = self._embedder.encode([claim])[0].tolist()

            templates = await db.templates.find(
                {},
                {"template_text": 1, "verdict": 1,
                 "credibility_score": 1, "explanation": 1,
                 "source": 1, "embedding": 1}
            ).to_list(length=200)

            for t in templates:
                if not t.get("embedding"):
                    continue
                a   = np.array(emb)
                b   = np.array(t["embedding"])
                sim = float(
                    np.dot(a, b) /
                    (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                )
                if sim > 0.85:
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
                        "web_searched":      False,
                    }
        except Exception as e:
            logger.warning(f"Template cache check failed: {e}")
        return None

    # ── Main Groq call with tool calling ─────────────────────

    async def _call_groq_with_tools(
        self, claim: str, faiss_docs: list
    ) -> dict:
        """
        Call Groq with web search tool.
        Groq decides whether to search the web based on evidence quality.
        """
        docs_context  = self._format_docs(faiss_docs)
        user_message  = build_user_prompt(claim, docs_context)
        from typing import Any
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
        web_searched  = False
        search_results = []

        # ── Round 1: Initial Groq call ────────────────────────
        response = self._groq.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
            tool_choice="auto",  # Groq decides when to search
            temperature=0.1,
            max_tokens=1024,
        )

        msg = response.choices[0].message

        # ── Check if Groq wants to search the web ─────────────
        if msg.tool_calls:
            logger.info(f"Groq requesting {len(msg.tool_calls)} web search(es)")

            # Add assistant message to conversation
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            # Execute each tool call
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "search_web":
                    args  = json.loads(tool_call.function.arguments)
                    query = args.get("query", claim)

                    # Fast path: check recent RSS first
                    rss_results = []
                    try:
                        from backend.core.rss_monitor import get_rss_monitor
                        monitor     = await get_rss_monitor()
                        rss_results = await monitor.search_recent_rss(query)
                        if rss_results:
                            logger.info(
                                f"RSS fast-path: {len(rss_results)} "
                                f"recent entries found"
                            )
                    except Exception:
                        pass

                    # If RSS has good results use them, otherwise web search
                    if rss_results and rss_results[0]["similarity"] > 0.5:
                        results = rss_results
                        logger.info("Using RSS results (skipping web search)")
                    else:
                        results = await self._web_search(query)
                        logger.info("Using web search results")

                    search_results.extend(results)
                    web_searched = True

                    # Format results for Groq
                    search_text = "\n\n".join([
                        f"[Web Result {i+1}] {r['url']}\n{r['text'][:300]}"
                        for i, r in enumerate(results)
                    ]) if results else "No web results found."

                    # Add tool result to conversation
                    messages.append({
                        "role":        "tool",
                        "tool_call_id": tool_call.id,
                        "content":     search_text,
                    })

            # ── Round 2: Groq reasons with web results ────────
            logger.info("Groq reasoning with web search results...")
            response2 = self._groq.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            final_text = response2.choices[0].message.content or ""

        else:
            # Groq decided FAISS docs were sufficient — no web search
            logger.info("Groq using FAISS docs only (no web search needed)")
            final_text = msg.content or ""

        # ── Parse verdict JSON ────────────────────────────────
        result = self._parse_verdict(final_text, faiss_docs + search_results)
        result["web_searched"]    = web_searched
        result["web_result_count"] = len(search_results)

        return result

    def _parse_verdict(self, raw: str, docs: list) -> dict:
        """Extract JSON verdict from LLM output."""
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
            "sources":           [d["url"] for d in docs[:2]],
        }

    # ── Main entry point ──────────────────────────────────────

    async def verify(
        self, claim: str, platform: str = "manual"
    ) -> dict:
        if not self._ready:
            await self.initialize()

        logger.info(f"Verifying: {claim[:80]}...")

        # Step 1 — classify claim type
        claim_type = self._classify_claim_type(claim)

        # Step 2 — check template cache (instant path)
        cached = await self._check_template_cache(claim)
        if cached:
            logger.info("Template cache hit ✓")
            return {
                **cached,
                "claim":          claim,
                "claim_type":     claim_type,
                "platform":       platform,
                "retrieved_docs": [],
                "from_cache":     True,
                "verified_at":    datetime.utcnow().isoformat(),
            }

        # Step 3 — FAISS retrieval
        retrieved = self._retrieve(claim, claim_type)
        top_sim = retrieved[0]['similarity'] if retrieved else 0.0
        logger.info(
            f"FAISS retrieved {len(retrieved)} docs, "
            f"top similarity: {top_sim:.3f}"
        )

        # Step 4 — Groq with tool calling
        # (Groq decides whether to search web based on evidence quality)
        try:
            result = await self._call_groq_with_tools(claim, retrieved)
        except Exception as e:
            logger.error(f"Groq call failed: {e}")
            return self._error_result(claim, claim_type, str(e))

        # Step 5 — trust weight adjustment
        all_docs = retrieved + ([] if not result.get("web_searched") else [])
        if all_docs:
            top_w = max(
                (d.get("final_weight", d.get("weight", 0.6))
                 for d in all_docs[:3]),
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
            f"Web searched: {result.get('web_searched', False)}"
        )
        return final

    # ── Helpers ───────────────────────────────────────────────

    def _error_result(
        self, claim: str, claim_type: str, error: str
    ) -> dict:
        return {
            "claim":             claim,
            "verdict":           "UNVERIFIABLE",
            "credibility_score": 0,
            "reasoning":         f"Verification error: {error}",
            "sources":           [],
            "claim_type":        claim_type,
            "retrieved_docs":    [],
            "web_searched":      False,
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