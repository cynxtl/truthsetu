import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import feedparser
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.db.mongodb import get_db

settings = get_settings()

# ── Crisis keywords ───────────────────────────────────────────
CRISIS_KEYWORDS = [
    # English — weather & disaster
    "cyclone", "flood", "earthquake", "tsunami", "disaster",
    "evacuation", "emergency", "alert", "warning", "rescue",
    "trapped", "missing", "casualties", "death toll", "blast",
    "explosion", "fire", "attack", "riot", "violence",
    # English — health
    "pandemic", "virus", "outbreak", "epidemic", "covid",
    "corona", "vaccine", "vaccination", "injection", "dose",
    "booster", "lockdown", "quarantine", "hospital", "medicine",
    "drug", "cure", "treatment", "health emergency", "disease",
    # English — political/social
    "election fraud", "evm hacked", "fake news", "misinformation",
    "free distribution", "government scheme", "aadhaar", "ban",
    "demonetization", "currency", "arrest", "curfew",
    # English — infrastructure
    "dam broke", "dam burst", "radiation leak", "gas leak",
    "power cut", "water supply", "contaminated",
    # Hindi
    "बाढ़", "चक्रवात", "भूकंप", "महामारी", "वायरस",
    "अफवाह", "निकासी", "चुनाव", "बांध टूटा", "टीका",
    "लॉकडाउन", "अस्पताल", "दवा", "मौत", "हमला",
    # Tamil
    "வெள்ளம்", "புயல்", "நிலநடுக்கம்", "தொற்று", "தடுப்பூசி",
    # Telugu
    "వరద", "తుఫాను", "భూకంపం", "వైరస్", "వ్యాక్సిన్",
    # Marathi
    "पूर", "चक्रीवादळ", "भूकंप", "साथीचा रोग", "लस",
    # Bengali
    "বন্যা", "ঘূর্ণিঝড়", "ভূমিকম্প", "ভাইরাস", "ভ্যাকসিন",
]

# ── Virality thresholds ───────────────────────────────────────
VIRALITY_THRESHOLDS = {
    "twitter":  {"retweets": 50,  "replies": 20,  "likes": 200},
    "telegram": {"forwards": 30},
    "whatsapp": {"reports": 3},   # 3 different users report same claim
    "manual":   {"reports": 1},   # manual submissions always pass
}


class ScoutAgent:

    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self._ready = False

    async def initialize(self):
        logger.info("Initialising SCOUT agent...")
        self._embedder = SentenceTransformer(settings.embedding_model)
        self._ready = True
        logger.success("SCOUT agent ready ✓")

    # ── Layer 1: Keyword filter ───────────────────────────────
    def _passes_keyword_filter(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in CRISIS_KEYWORDS)

    # ── Layer 2: Virality gate ────────────────────────────────
    def _passes_virality_gate(
        self, platform: str, metrics: dict
    ) -> bool:
        if platform == "manual":
            return True
        if platform == "whatsapp":
            return metrics.get("reports", 1) >= \
                   VIRALITY_THRESHOLDS["whatsapp"]["reports"]
        if platform == "twitter":
            t = VIRALITY_THRESHOLDS["twitter"]
            return (
                metrics.get("retweets", 0) >= t["retweets"] or
                metrics.get("replies",  0) >= t["replies"]  or
                metrics.get("likes",    0) >= t["likes"]
            )
        if platform == "telegram":
            return metrics.get("forwards", 0) >= \
                   VIRALITY_THRESHOLDS["telegram"]["forwards"]
        return True

    # ── Layer 3: Semantic deduplication ──────────────────────
    async def _is_duplicate(self, claim: str) -> Optional[dict]:
        try:
            db  = get_db()
            emb = self._embedder.encode([claim])[0].tolist()

            # Check claims from last 24 hours
            cutoff = datetime.utcnow() - timedelta(hours=24)
            recent = await db.claims.find(
                {"detected_at": {"$gte": cutoff}},
                {"claim_text": 1, "embedding": 1, "_id": 1}
            ).to_list(length=500)

            for doc in recent:
                if not doc.get("embedding"):
                    continue
                a   = np.array(emb)
                b   = np.array(doc["embedding"])
                sim = float(
                    np.dot(a, b) /
                    (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                )
                if sim > 0.85:
                    return {
                        "original_id":   str(doc["_id"]),
                        "original_claim": doc["claim_text"],
                        "similarity":     sim,
                    }
        except Exception as e:
            logger.warning(f"Dedup check failed: {e}")
        return None

    # ── Layer 4: Claim extraction via Groq ───────────────────
    async def _extract_claim(self, text: str) -> Optional[str]:
        try:
            from backend.core.llm import get_llm
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = PromptTemplate(
                input_variables=["text"],
                template="""Does this post contain a specific factual claim 
about a crisis, disaster, health emergency, or election?

If YES: Extract the single core factual claim in one clear sentence.
If NO:  Respond with exactly: NO_CLAIM

Post: "{text}"

Respond with either the claim sentence or NO_CLAIM only:"""
            )
            chain  = prompt | get_llm() | StrOutputParser()
            result = await chain.ainvoke({"text": text[:500]})
            result = result.strip()

            if result == "NO_CLAIM" or len(result) < 10:
                return None
            return result

        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            return None

    # ── Main entry point ─────────────────────────────────────
    async def process(
        self,
        text: str,
        platform: str = "manual",
        sender_number: Optional[str] = None,
        source_url: Optional[str] = None,
        metrics: Optional[dict] = None,
    ) -> dict:
        if not self._ready:
            await self.initialize()

        metrics = metrics or {}
        logger.info(f"SCOUT processing [{platform}]: {text[:60]}...")

        # Layer 1 — keyword filter
        if not self._passes_keyword_filter(text):
            logger.debug("Dropped: no crisis keywords")
            return {"status": "dropped", "reason": "no_crisis_keywords"}

        # Layer 2 — virality gate (manual submissions always pass)
        if not self._passes_virality_gate(platform, metrics):
            logger.debug("Dropped: below virality threshold")
            return {"status": "dropped", "reason": "below_virality_threshold"}

        # Layer 3 — extract claim via LLM
        claim = await self._extract_claim(text)
        if not claim:
            logger.debug("Dropped: no factual claim found")
            return {"status": "dropped", "reason": "no_claim_found"}

        # Layer 4 — semantic deduplication
        duplicate = await self._is_duplicate(claim)
        if duplicate:
            logger.info(f"Duplicate detected (sim={duplicate['similarity']:.2f})")
            return {
                "status":    "duplicate",
                "reason":    "duplicate_claim",
                "duplicate": duplicate,
            }

        # Save to MongoDB
        emb = self._embedder.encode([claim])[0].tolist()
        db  = get_db()
        doc = {
            "claim_text":    claim,
            "original_text": text,
            "platform":      platform,
            "sender_number": sender_number,
            "source_url":    source_url,
            "embedding":     emb,
            "detected_at":   datetime.utcnow(),
            "status":        "queued",
        }
        result = await db.claims.insert_one(doc)
        claim_id = str(result.inserted_id)

        logger.success(f"New claim saved: {claim_id}")
        return {
            "status":   "queued",
            "claim_id": claim_id,
            "claim":    claim,
            "platform": platform,
        }

    # ── WhatsApp incoming handler ─────────────────────────────
    async def handle_whatsapp(
        self, message: str, sender_number: str
    ) -> dict:
        return await self.process(
            text=message,
            platform="whatsapp",
            sender_number=sender_number,
            metrics={"reports": 1},
        )

    # ── Status ────────────────────────────────────────────────
    async def get_status(self) -> dict:
        try:
            db    = get_db()
            total = await db.claims.count_documents({})
            today = await db.claims.count_documents({
                "detected_at": {
                    "$gte": datetime.utcnow() - timedelta(hours=24)
                }
            })
            return {
                "ready":        self._ready,
                "monitoring":   True,
                "claims_total": total,
                "claims_today": today,
            }
        except Exception:
            return {"ready": self._ready, "monitoring": False}


# ── Singleton ─────────────────────────────────────────────────
_scout: Optional[ScoutAgent] = None


async def get_scout_agent() -> ScoutAgent:
    global _scout
    if _scout is None:
        _scout = ScoutAgent()
        await _scout.initialize()
    return _scout