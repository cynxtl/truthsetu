"""
TruthSetu — Pipeline Orchestrator
SCOUT → VERIFY → TRANSLATE → DEPLOY → LEARN
"""
import asyncio
from datetime import datetime, timezone
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
        if not self._scout:
            from backend.agents.scout_agent import get_scout_agent
            self._scout = await get_scout_agent()

        if not self._verify:
            from backend.agents.verify_agent import get_verify_agent
            self._verify = await get_verify_agent()

        if not self._translate:
            from backend.agents.translate_agent import get_translate_agent
            self._translate = await get_translate_agent()

        if not self._deploy:
            from backend.agents.deploy_agent import get_deploy_agent
            self._deploy = await get_deploy_agent()

        if not self._learn:
            from backend.agents.learn_agent import get_learn_agent
            self._learn = await get_learn_agent()

    async def run(
        self,
        text: str,
        platform: str = "manual",
        sender_number: Optional[str] = None,
        source_url: Optional[str] = None,
        metrics: Optional[dict] = None,
        language: str = "en",
    ) -> dict:
        await self._get_agents()

        pipeline_start = datetime.now(timezone.utc)
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
                result["status"]     = "dropped"
                result["reason"]     = scout_result["reason"]
                result["stopped_at"] = "scout"

                # Send helpful reply to citizen
                if sender_number and scout_result["reason"] == "no_claim_found":
                    try:
                        deploy = await self._deploy._send_whatsapp(
                            sender_number,
                            "🤔 *We couldn't find a verifiable claim in your message.*\n\n"
                            "Please try sending:\n"
                            "• A specific claim to check\n"
                            "• A question about news or events\n"
                            "• A WhatsApp forward you want verified\n\n"
                            "Example: _\"Is it true that cyclone is hitting Chennai?\"_\n\n"
                            "— TruthSetu"
                        )
                    except Exception:
                        pass

                return result

            claim = scout_result["duplicate"]["original_claim"] \
                if scout_result["status"] == "duplicate" \
                else scout_result["claim"]

            if scout_result["status"] == "duplicate":
                result["steps"]["scout"]["extracted_claim"] = claim

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
                "credibility_score": verify_result.get("credibility_score", 0),
                "reasoning":         verify_result["reasoning"],
                "sources":           verify_result["sources"],
                "from_cache":        verify_result.get("from_cache", False),
                "claim_type":        verify_result.get("claim_type", "EPHEMERAL"),
            }

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
                reasoning=verify_result["reasoning"],
                sources=verify_result["sources"],
            )

            # Detect citizen's language from original message
            detected_lang = await self._translate.detect_language(text)
            logger.info(f"Detected citizen language: {detected_lang}")

            # Translate only if non-English
            if detected_lang in ["hi", "mr"]:
                translated = await self._translate.translate(
                    message=correction_en,
                    target_languages=[detected_lang],
                    citizen_language=detected_lang,
                )
            else:
                # English or unknown → reply in English
                translated = {
                    "translations":      {"en": {"text": correction_en, "success": True}},
                    "priority_language": "en",
                }
                detected_lang = "en"

            result["steps"]["translate"] = {
                "correction_en":     correction_en,
                "translations":      translated["translations"],
                "priority_language": translated["priority_language"],
                "citizen_language":  detected_lang,
            }
            language = detected_lang

        except Exception as e:
            logger.warning(f"TRANSLATE failed (non-fatal): {e}")
            correction_en = self._compose_correction(
                claim=claim,
                verdict=verify_result["verdict"],
                reasoning=verify_result["reasoning"],
                sources=verify_result["sources"],
            )
            result["steps"]["translate"] = {
                "correction_en":     correction_en,
                "translations":      {"en": {"text": correction_en, "success": True}},
                "priority_language": "en",
                "citizen_language":  "en",
            }
            language = "en"

        # ── Step 4: DEPLOY ────────────────────────────────────
        logger.info("Step 4: DEPLOY")
        try:
            translations  = result["steps"]["translate"].get("translations", {})
            deploy_result = await self._deploy.deploy(
                sender_number=sender_number or "",
                claim=claim,
                verdict=verify_result["verdict"],
                score=verify_result.get("credibility_score", 0),
                reasoning=verify_result["reasoning"],
                sources=verify_result["sources"],
                translations=translations,
                citizen_language=language,
                platform=platform,
            )
            result["steps"]["deploy"] = deploy_result

        except Exception as e:
            logger.warning(f"DEPLOY failed (non-fatal): {e}")
            result["steps"]["deploy"] = {"status": "failed", "error": str(e)}

        # ── Step 5: LEARN ─────────────────────────────────────────
        logger.info("Step 5: LEARN")
        try:
            learn_result = await self._learn.learn(
                claim=claim,
                verdict=verify_result["verdict"],
                reasoning=verify_result["reasoning"],
                sources=verify_result["sources"],
                platform=platform,
                claim_type=verify_result.get("claim_type", "EPHEMERAL"),
            )
            result["steps"]["learn"] = learn_result
        except Exception as e:
            logger.warning(f"LEARN failed (non-fatal): {e}")
            result["steps"]["learn"] = {"status": "failed", "error": str(e)}

        # ── Final ─────────────────────────────────────────────
        pipeline_end = datetime.now(timezone.utc)
        duration     = (pipeline_end - pipeline_start).total_seconds()

        result["status"]           = "completed"
        result["verdict"]          = verify_result["verdict"]
        result["completed_at"]     = pipeline_end.isoformat()
        result["duration_seconds"] = round(duration, 2)

        logger.success(
            f"Pipeline completed in {duration:.1f}s | "
            f"Verdict: {verify_result['verdict']}"
        )
        return result

    def _compose_correction(
        self,
        claim:     str,
        verdict:   str,
        reasoning: str,
        sources:   list,
    ) -> str:
        """Compose the WhatsApp reply — no score, just summary."""
        source_line = f"\n📎 Source: {sources[0]}" if sources else ""
        timestamp   = datetime.now(timezone.utc).strftime(
            "%d %b %Y, %I:%M %p"
        ) + " IST"

        if verdict == "FALSE":
            return (
                f"🔴 *FACT CHECK — FALSE*\n\n"
                f"❌ Claim: \"{claim}\"\n\n"
                f"✅ What sources say:\n{reasoning}"
                f"{source_line}\n\n"
                f"⏱ Checked: {timestamp}\n"
                f"— TruthSetu"
            )
        elif verdict in ["TRUE", "CONFIRMED"]:
            return (
                f"🟢 *FACT CHECK — CONFIRMED*\n\n"
                f"✅ Claim: \"{claim}\"\n\n"
                f"📋 What sources say:\n{reasoning}"
                f"{source_line}\n\n"
                f"⏱ Checked: {timestamp}\n"
                f"— TruthSetu"
            )
        else:
            return (
                f"🟡 *FACT CHECK — UNVERIFIED*\n\n"
                f"⚠️ Claim: \"{claim}\"\n\n"
                f"📋 What sources say:\n{reasoning}"
                f"{source_line}\n\n"
                f"⏱ Checked: {timestamp}\n"
                f"— TruthSetu"
            )

    async def _learn_stub(self, claim: str, verdict: dict) -> None:
        logger.info(f"[LEARN STUB] Would store template for: {claim[:60]}")


_pipeline: Optional[TruthSetuPipeline] = None


async def get_pipeline() -> TruthSetuPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TruthSetuPipeline()
    return _pipeline