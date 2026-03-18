"""
TruthSetu — Pipeline Orchestrator
Connects all 5 agents end to end using simple async chain.

Flow:
  SCOUT → VERIFY → TRANSLATE → DEPLOY → LEARN

Called from:
  - WhatsApp webhook (citizen forwards message)
  - Manual submission via /scout/submit
  - Admin dashboard
"""
import asyncio
from datetime import datetime
from typing import Optional
from loguru import logger

from backend.db.mongodb import get_db


class TruthSetuPipeline:

    def __init__(self):
        self._scout     = None
        self._verify    = None
        self._translate = None
        self._deploy    = None
        self._learn     = None

    async def _get_agents(self):
        """Lazy load agents."""
        if not self._scout:
            from backend.agents.scout_agent import get_scout_agent
            self._scout = await get_scout_agent()

        if not self._verify:
            from backend.agents.verify_agent import get_verify_agent
            self._verify = await get_verify_agent()

        if not self._translate:
            from backend.agents.translate_agent import get_translate_agent
            self._translate = await get_translate_agent()


    async def run(
        self,
        text: str,
        platform: str = "manual",
        sender_number: Optional[str] = None,
        source_url: Optional[str] = None,
        metrics: Optional[dict] = None,
        language: str = "en",
    ) -> dict:
        """
        Main entry point. Run full pipeline for a single message.
        Returns final result dict.
        """
        await self._get_agents()

        pipeline_start = datetime.utcnow()
        logger.info(f"Pipeline started [{platform}]: {text[:60]}...")

        result = {
            "input_text":    text,
            "platform":      platform,
            "sender_number": sender_number,
            "language":      language,
            "started_at":    pipeline_start.isoformat(),
            "steps":         {},
        }

        # ── Step 1: SCOUT ─────────────────────────────────────
        logger.info("Step 1: SCOUT")
        try:
            scout_result = await self._scout.process(
                text=text,
                platform=platform,
                sender_number=sender_number,
                source_url=source_url,
                metrics=metrics or {},
            )
            result["steps"]["scout"] = scout_result

            if scout_result["status"] == "dropped":
                logger.info(f"Pipeline stopped at SCOUT: {scout_result['reason']}")
                result["status"]  = "dropped"
                result["reason"]  = scout_result["reason"]
                result["stopped_at"] = "scout"
                return result

            if scout_result["status"] == "duplicate":
                # Duplicate — still send cached verdict to citizen
                logger.info("Duplicate claim — fetching cached verdict")
                claim = scout_result["duplicate"]["original_claim"]
                result["steps"]["scout"]["extracted_claim"] = claim
                # Skip to VERIFY with the original claim
                # (will hit template cache instantly)
            else:
                claim = scout_result["claim"]

        except Exception as e:
            logger.error(f"SCOUT failed: {e}")
            result["status"]     = "error"
            result["error"]      = str(e)
            result["stopped_at"] = "scout"
            return result

        # ── Step 2: VERIFY ────────────────────────────────────
        logger.info("Step 2: VERIFY")
        try:
            verify_result = await self._verify.verify(claim, platform)
            result["steps"]["verify"] = {
                "verdict":           verify_result["verdict"],
                "credibility_score": verify_result["credibility_score"],
                "reasoning":         verify_result["reasoning"],
                "sources":           verify_result["sources"],
                "from_cache":        verify_result.get("from_cache", False),
                "claim_type":        verify_result.get("claim_type", "EPHEMERAL"),
            }

            # Save verdict to MongoDB
            try:
                db = get_db()
                await db.verdicts.insert_one({
                    **{k: v for k, v in verify_result.items()
                       if k != "retrieved_docs"},
                    "sender_number": sender_number,
                    "pipeline_run":  True,
                })
            except Exception:
                pass

        except Exception as e:
            logger.error(f"VERIFY failed: {e}")
            result["status"]     = "error"
            result["error"]      = str(e)
            result["stopped_at"] = "verify"
            return result

        # ── Step 3: TRANSLATE ─────────────────────────────────
        logger.info("Step 3: TRANSLATE")
        try:
            correction_en = self._compose_correction(
                claim=claim,
                verdict=verify_result["verdict"],
                score=verify_result["credibility_score"],
                reasoning=verify_result["reasoning"],
                sources=verify_result["sources"],
            )

            # TRANSLATE agent not built yet — use English for now
            # Will be replaced when translate_agent.py is built
            # Detect citizen language if not provided
            if language == "en":
                language = await self._translate.detect_language(text)
                logger.info(f"Detected citizen language: {language}")

            # Translate to Hindi + Marathi always
            # Plus citizen's language if different
            target_langs = ["hi", "mr"]
            if language not in target_langs and language != "en":
                target_langs.append(language)

            translated = await self._translate.translate(
                message=correction_en,
                target_languages=target_langs,
                citizen_language=language,
            )
            result["steps"]["translate"] = {
                "correction_en":     correction_en,
                "translations":      translated["translations"],
                "priority_language": translated["priority_language"],
                "citizen_language":  language,
            }

        except Exception as e:
            logger.warning(f"TRANSLATE failed (non-fatal): {e}")
            correction_en = self._compose_correction(
                claim=claim,
                verdict=verify_result["verdict"],
                score=verify_result["credibility_score"],
                reasoning=verify_result["reasoning"],
                sources=verify_result["sources"],
            )
            result["steps"]["translate"] = {
                "correction_en": correction_en,
                "translations":  {"en": correction_en},
            }

        # ── Step 4: DEPLOY ────────────────────────────────────
        logger.info("Step 4: DEPLOY")
        try:
            deploy_result = await self._deploy_stub(
                sender_number=sender_number,
                platform=platform,
                message=result["steps"]["translate"]["correction_en"],
                language=language,
            )
            result["steps"]["deploy"] = deploy_result

        except Exception as e:
            logger.warning(f"DEPLOY failed (non-fatal): {e}")
            result["steps"]["deploy"] = {"status": "failed", "error": str(e)}

        # ── Step 5: LEARN ─────────────────────────────────────
        logger.info("Step 5: LEARN")
        try:
            if verify_result["verdict"] == "FALSE":
                await self._learn_stub(claim, verify_result)
                result["steps"]["learn"] = {"status": "template_stored"}
            else:
                result["steps"]["learn"] = {"status": "skipped"}

        except Exception as e:
            logger.warning(f"LEARN failed (non-fatal): {e}")
            result["steps"]["learn"] = {"status": "failed", "error": str(e)}

        # ── Final result ──────────────────────────────────────
        pipeline_end = datetime.utcnow()
        duration     = (pipeline_end - pipeline_start).total_seconds()

        result["status"]      = "completed"
        result["verdict"]     = verify_result["verdict"]
        result["score"]       = verify_result["credibility_score"]
        result["completed_at"] = pipeline_end.isoformat()
        result["duration_seconds"] = round(duration, 2)

        logger.success(
            f"Pipeline completed in {duration:.1f}s | "
            f"Verdict: {verify_result['verdict']} | "
            f"Score: {verify_result['credibility_score']}"
        )
        return result

    # ── Message composer ──────────────────────────────────────
    def _compose_correction(
        self,
        claim: str,
        verdict: str,
        score: int,
        reasoning: str,
        sources: list,
    ) -> str:
        """Compose the WhatsApp reply message."""

        source_line = ""
        if sources:
            source_line = f"\n📎 Source: {sources[0]}"

        if verdict == "FALSE":
            return (
                f"🔴 *FACT CHECK — FALSE*\n\n"
                f"❌ Claim being shared:\n\"{claim}\"\n\n"
                f"✅ What is actually true:\n{reasoning}"
                f"{source_line}\n\n"
                f"📊 Credibility: {score}/100\n"
                f"⏱ Checked: {datetime.utcnow().strftime('%d %b %Y, %I:%M %p')} IST\n\n"
                f"— TruthSetu | Verified Information"
            )

        elif verdict == "TRUE":
            return (
                f"🟢 *FACT CHECK — CONFIRMED*\n\n"
                f"✅ This claim is confirmed:\n\"{claim}\"\n\n"
                f"{reasoning}"
                f"{source_line}\n\n"
                f"📊 Credibility: {score}/100\n"
                f"⏱ Checked: {datetime.utcnow().strftime('%d %b %Y, %I:%M %p')} IST\n\n"
                f"— TruthSetu | Verified Information"
            )

        else:  # UNVERIFIABLE
            return (
                f"🟡 *FACT CHECK — UNVERIFIABLE*\n\n"
                f"⚠️ We could not confirm or deny:\n\"{claim}\"\n\n"
                f"No official sources have addressed this yet.\n"
                f"Please check:\n"
                f"• ndma.gov.in\n"
                f"• mausam.imd.gov.in\n"
                f"• mohfw.gov.in\n\n"
                f"⏱ Checked: {datetime.utcnow().strftime('%d %b %Y, %I:%M %p')} IST\n\n"
                f"— TruthSetu | Verified Information"
            )

    # ── Stubs (replaced when agents are built) ────────────────
    async def _translate_stub(
        self, text: str, language: str
    ) -> dict:
        """
        Placeholder until translate_agent.py is built.
        Returns English only for now.
        """
        return {"en": text}

    async def _deploy_stub(
        self,
        sender_number: Optional[str],
        platform: str,
        message: str,
        language: str,
    ) -> dict:
        """
        Placeholder until deploy_agent.py is built.
        Logs the message that would be sent.
        """
        if sender_number:
            logger.info(
                f"[DEPLOY STUB] Would send to {sender_number} "
                f"via {platform}:\n{message[:100]}..."
            )
            return {
                "status":   "stub",
                "would_send_to": sender_number,
                "platform": platform,
                "message_preview": message[:100],
            }
        return {"status": "no_recipient"}

    async def _learn_stub(self, claim: str, verdict: dict) -> None:
        """
        Placeholder until learn_agent.py is built.
        Just logs for now.
        """
        logger.info(
            f"[LEARN STUB] Would store template for: {claim[:60]}"
        )


# ── Singleton ─────────────────────────────────────────────────
_pipeline: Optional[TruthSetuPipeline] = None


async def get_pipeline() -> TruthSetuPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TruthSetuPipeline()
    return _pipeline