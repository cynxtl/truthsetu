"""
TruthSetu — LEARN Agent
========================
Learns from every verified claim to make future verifications faster.

Three jobs:
  1. Store FALSE verdicts as reusable templates
  2. Update source quality scores based on accuracy
  3. Pre-emptive seeding — extract templates from fact-checker articles
"""

from datetime import datetime, timezone
from typing import Optional
from loguru import logger

from backend.core.config import get_settings

settings = get_settings()


class LearnAgent:

    def __init__(self):
        self._embedder = None
        self._ready    = False

    async def initialize(self):
        logger.info("Initialising LEARN agent...")
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(settings.embedding_model)
        self._ready    = True
        logger.success("LEARN agent ready ✓")

    # ── Main entry point ──────────────────────────────────────

    async def learn(
        self,
        claim:        str,
        verdict:      str,
        reasoning:    str,
        sources:      list,
        platform:     str = "manual",
        claim_type:   str = "EPHEMERAL",
    ) -> dict:
        """
        Called after every verified claim.
        Stores templates for FALSE verdicts.
        Updates source scores for all verdicts.
        """
        if not self._ready:
            await self.initialize()

        results = {}

        # Always log to training data
        await self._store_training_data(
            claim=claim,
            verdict=verdict,
            reasoning=reasoning,
            sources=sources,
            platform=platform,
        )
        results["training_data"] = "stored"

        # Store template only for FALSE verdicts
        if verdict == "FALSE":
            template_id = await self._store_template(
                claim=claim,
                reasoning=reasoning,
                sources=sources,
                claim_type=claim_type,
            )
            results["template"] = template_id
            logger.info(f"LEARN: stored FALSE template for: {claim[:60]}")
        else:
            results["template"] = "skipped"

        # Update source quality scores
        await self._update_source_scores(sources, verdict)
        results["source_scores"] = "updated"

        return results

    # ── Store template ────────────────────────────────────────

    async def _store_template(
        self,
        claim:      str,
        reasoning:  str,
        sources:    list,
        claim_type: str,
    ) -> Optional[str]:
        """
        Store a FALSE claim as a template for instant future cache hits.
        Next time a similar claim arrives → returns FALSE instantly
        without calling Groq.
        """
        try:
            from backend.db.mongodb import get_db
            db = get_db()

            # Generate embedding
            emb = self._embedder.encode([claim])[0].tolist()

            # Check if very similar template already exists
            existing = await self._find_similar_template(claim, emb, threshold=0.92)
            if existing:
                # Update occurrence count instead of creating duplicate
                await db.templates.update_one(
                    {"_id": existing["_id"]},
                    {
                        "$inc": {"occurrence_count": 1},
                        "$set": {"last_seen": datetime.now(timezone.utc)},
                    }
                )
                logger.debug(
                    f"LEARN: updated existing template "
                    f"(count={existing.get('occurrence_count', 1)+1})"
                )
                return str(existing["_id"])

            # Detect crisis type
            crisis_type = self._detect_crisis_type(claim)

            # Store new template
            result = await db.templates.insert_one({
                "template_text":    claim,
                "example_claim":    claim,
                "crisis_type":      crisis_type,
                "claim_type":       claim_type,
                "verdict":          "FALSE",
                "credibility_score": 5,
                "source":           sources[0] if sources else "web_search",
                "explanation":      reasoning,
                "embedding":        emb,
                "first_seen":       datetime.now(timezone.utc),
                "last_seen":        datetime.now(timezone.utc),
                "occurrence_count": 1,
                "auto_learned":     True,
            })

            logger.success(
                f"LEARN: new template stored [{crisis_type}]: "
                f"{claim[:60]}"
            )
            return str(result.inserted_id)

        except Exception as e:
            logger.warning(f"LEARN: template storage failed: {e}")
            return None

    async def _find_similar_template(
        self, claim: str, emb: list, threshold: float = 0.92
    ) -> Optional[dict]:
        """Check if a very similar template already exists."""
        try:
            import numpy as np
            from backend.db.mongodb import get_db
            db = get_db()

            templates = await db.templates.find(
                {"auto_learned": True},
                {"template_text": 1, "embedding": 1,
                 "occurrence_count": 1, "_id": 1}
            ).to_list(length=500)

            a = np.array(emb)
            for t in templates:
                if not t.get("embedding"):
                    continue
                b   = np.array(t["embedding"])
                sim = float(
                    np.dot(a, b) /
                    (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                )
                if sim > threshold:
                    return t

        except Exception as e:
            logger.debug(f"LEARN: similar template check failed: {e}")
        return None

    # ── Store training data ───────────────────────────────────

    async def _store_training_data(
        self,
        claim:    str,
        verdict:  str,
        reasoning: str,
        sources:  list,
        platform: str,
    ):
        """
        Store every verified claim as training data.
        Used later for fine-tuning or classifier training.
        """
        try:
            from backend.db.mongodb import get_db
            db  = get_db()
            emb = self._embedder.encode([claim])[0].tolist()

            await db.training_data.insert_one({
                "claim":     claim,
                "verdict":   verdict,
                "reasoning": reasoning,
                "sources":   sources,
                "platform":  platform,
                "embedding": emb,
                "date":      datetime.now(timezone.utc),
            })

        except Exception as e:
            logger.debug(f"LEARN: training data storage failed: {e}")

    # ── Update source scores ──────────────────────────────────

    async def _update_source_scores(
        self, sources: list, verdict: str
    ):
        """
        Track how often each source is cited and for which verdict.
        Builds a quality picture of each source over time.
        """
        try:
            from backend.db.mongodb import get_db
            db = get_db()

            for source in sources:
                if not source or source == "web_search":
                    continue

                # Extract domain from URL if needed
                domain = source
                if source.startswith("http"):
                    from urllib.parse import urlparse
                    domain = urlparse(source).netloc.replace("www.", "")

                await db.source_scores.update_one(
                    {"source": domain},
                    {
                        "$inc": {
                            "total_citations": 1,
                            f"verdict_{verdict.lower()}_count": 1,
                        },
                        "$set": {
                            "last_cited": datetime.now(timezone.utc),
                        },
                        "$setOnInsert": {
                            "source":     domain,
                            "first_seen": datetime.now(timezone.utc),
                            "weight":     0.75,
                            "tier":       "unknown",
                        }
                    },
                    upsert=True
                )

        except Exception as e:
            logger.debug(f"LEARN: source score update failed: {e}")

    # ── Detect crisis type ────────────────────────────────────

    def _detect_crisis_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in [
            "cyclone", "hurricane", "typhoon", "storm", "flood",
            "rain", "dam", "river", "earthquake", "tremor"
        ]):
            return "natural_disaster"
        if any(w in text_lower for w in [
            "covid", "corona", "pandemic", "virus", "vaccine",
            "outbreak", "epidemic", "monkeypox", "disease"
        ]):
            return "health"
        if any(w in text_lower for w in [
            "election", "evm", "vote", "ballot", "polling",
            "candidate", "modi", "rahul", "government"
        ]):
            return "political"
        if any(w in text_lower for w in [
            "war", "attack", "riot", "communal", "violence",
            "explosion", "blast", "bomb", "terror"
        ]):
            return "conflict"
        if any(w in text_lower for w in [
            "rbi", "rupee", "bank", "currency", "demonetization",
            "scheme", "policy", "law", "ban"
        ]):
            return "economic"
        return "general"

    # ── Stats ─────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        try:
            from backend.db.mongodb import get_db
            db            = get_db()
            templates     = await db.templates.count_documents({})
            auto_learned  = await db.templates.count_documents(
                {"auto_learned": True}
            )
            training_docs = await db.training_data.count_documents({})
            return {
                "ready":               self._ready,
                "total_templates":     templates,
                "auto_learned":        auto_learned,
                "training_data_docs":  training_docs,
            }
        except Exception:
            return {"ready": self._ready}


# ── Singleton ─────────────────────────────────────────────────
_agent: Optional[LearnAgent] = None


async def get_learn_agent() -> LearnAgent:
    global _agent
    if _agent is None:
        _agent = LearnAgent()
        await _agent.initialize()
    return _agent