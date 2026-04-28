"""
TruthSetu — SCOUT Agent
"""
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from loguru import logger
from backend.core.config import get_settings
from backend.db.mongodb import get_db

settings = get_settings()

VIRALITY_THRESHOLDS = {
    "twitter":  {"retweets": 50, "replies": 20, "likes": 200},
    "telegram": {"forwards": 30},
    "whatsapp": {"reports": 1},
    "manual":   {"reports": 1},
}


class ScoutAgent:

    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self._ready = False

    async def initialize(self):
        logger.info("Initialising SCOUT agent...")
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(settings.embedding_model)
        self._ready = True
        logger.success("SCOUT agent ready ✓")

    def _passes_virality_gate(self, platform: str, metrics: dict) -> bool:
        if platform in ["manual", "whatsapp"]:
            return True
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

    async def _is_duplicate(self, claim: str) -> Optional[dict]:
        try:
            db  = get_db()
            emb = self._embedder.encode([claim])[0].tolist()
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
                        "original_id":    str(doc["_id"]),
                        "original_claim": doc["claim_text"],
                        "similarity":     sim,
                    }
        except Exception as e:
            logger.warning(f"Dedup check failed: {e}")
        return None

    async def _extract_claim(self, text: str) -> Optional[str]:
        try:
            from groq import Groq
            from backend.core.config import get_settings
            settings = get_settings()
            client = Groq(api_key=settings.groq_api_key)

            prompt = f"""A citizen sent this message to a fact-checking service.
Extract the core factual claim they want verified.

Rules:
- If it's a question like "is X true?" → extract "X is true" as the claim
- If it's a statement like "X happened" → extract it directly
- If it's a forward like "Breaking: X" → extract the core claim
- If it's just a greeting or completely unrelated → respond: NO_CLAIM

Examples:
  "is modi dead?" → "Modi has died"
  "Is it true cyclone is hitting Chennai?" → "A cyclone is hitting Chennai"
  "hello" → NO_CLAIM

Message: "{text[:500]}"

Respond with the claim in one sentence or NO_CLAIM only:"""

            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            result = response.choices[0].message.content.strip()

            if result == "NO_CLAIM" or len(result) < 5:
                return None
            return result

        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            return None

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

        if not self._passes_virality_gate(platform, metrics):
            logger.debug("Dropped: below virality threshold")
            return {"status": "dropped", "reason": "below_virality_threshold"}

        claim = await self._extract_claim(text)
        if not claim:
            logger.debug("Dropped: no factual claim found")
            return {"status": "dropped", "reason": "no_claim_found"}

        duplicate = await self._is_duplicate(claim)
        if duplicate:
            logger.info(f"Duplicate detected (sim={duplicate['similarity']:.2f})")
            return {
                "status":    "duplicate",
                "reason":    "duplicate_claim",
                "duplicate": duplicate,
            }

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

    async def handle_whatsapp(self, message: str, sender_number: str) -> dict:
        return await self.process(
            text=message,
            platform="whatsapp",
            sender_number=sender_number,
            metrics={"reports": 1},
        )

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


_scout: Optional[ScoutAgent] = None


async def get_scout_agent() -> ScoutAgent:
    global _scout
    if _scout is None:
        _scout = ScoutAgent()
        await _scout.initialize()
    return _scout