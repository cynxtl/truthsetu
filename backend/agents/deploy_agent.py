"""
TruthSetu — DEPLOY Agent
=========================
Sends verified corrections to citizens via:
  Primary:  WhatsApp (Twilio Sandbox — free trial)
  Fallback: Simulation (logs message)

Future:
  Telegram broadcast channel
  Meta official WhatsApp Business API
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger

from backend.core.config import get_settings

settings = get_settings()


def format_whatsapp_message(
    claim:     str,
    verdict:   str,
    score:     int,
    reasoning: str,
    sources:   list,
) -> str:
    """Compose English WhatsApp message."""
    source_line = f"\n📎 Source: {sources[0]}" if sources else ""
    ist_tz      = timezone(timedelta(hours=5, minutes=30))
    timestamp   = datetime.now(ist_tz).strftime(
        "%d %b %Y, %I:%M %p"
    ) + " IST"

    if verdict == "FALSE":
        return (
            f"🔴 *FACT CHECK — FALSE*\n\n"
            f"❌ Claim being shared:\n\"{claim}\"\n\n"
            f"✅ What is actually true:\n{reasoning}"
            f"{source_line}\n\n"
            f"📊 Credibility: {score}/100\n"
            f"⏱ Checked: {timestamp}\n\n"
            f"— TruthSetu | Verified Information"
        )
    elif verdict == "TRUE":
        return (
            f"🟢 *FACT CHECK — CONFIRMED*\n\n"
            f"✅ This is confirmed:\n\"{claim}\"\n\n"
            f"{reasoning}"
            f"{source_line}\n\n"
            f"📊 Credibility: {score}/100\n"
            f"⏱ Checked: {timestamp}\n\n"
            f"— TruthSetu | Verified Information"
        )
    else:
        return (
            f"🟡 *FACT CHECK — UNVERIFIABLE*\n\n"
            f"⚠️ We could not confirm or deny:\n\"{claim}\"\n\n"
            f"No official sources have addressed this yet.\n"
            f"⏱ Checked: {timestamp}\n\n"
            f"— TruthSetu | Verified Information"
        )


class DeployAgent:

    def __init__(self):
        self._twilio = None
        self._ready  = False

    async def initialize(self):
        logger.info("Initialising DEPLOY agent...")

        if settings.twilio_account_sid and \
           settings.twilio_auth_token:
            try:
                from twilio.rest import Client
                self._twilio = Client(
                    settings.twilio_account_sid,
                    settings.twilio_auth_token,
                )
                logger.success("Twilio client ready ✓")
            except Exception as e:
                logger.warning(f"Twilio init failed: {e}")
        else:
            logger.warning(
                "Twilio not configured — messages will be simulated"
            )

        self._ready = True
        logger.success("DEPLOY agent ready ✓")

    # ── Main deploy ───────────────────────────────────────────

    async def deploy(
        self,
        sender_number:    str,
        claim:            str,
        verdict:          str,
        score:            int,
        reasoning:        str,
        sources:          list,
        translations:     Optional[dict] = None,
        citizen_language: str = "en",
        platform:         str = "whatsapp",
        verdict_id:       Optional[str] = None,
    ) -> dict:
        if not self._ready:
            await self.initialize()

        if not sender_number:
            return {"status": "skipped", "reason": "no_recipient"}

        # Pick best language message
        message = self._pick_message(
            claim=claim,
            verdict=verdict,
            score=score,
            reasoning=reasoning,
            sources=sources,
            translations=translations,
            citizen_language=citizen_language,
        )

        # Send
        result = await self._send_whatsapp(sender_number, message)

        # Log
        await self._log_deployment(
            verdict_id=verdict_id,
            sender_number=sender_number,
            platform=platform,
            message=message,
            language=citizen_language,
            result=result,
        )

        return result

    # ── Message picker ────────────────────────────────────────

    def _pick_message(
        self,
        claim:            str,
        verdict:          str,
        score:            int,
        reasoning:        str,
        sources:          list,
        translations:     Optional[dict],
        citizen_language: str,
    ) -> str:
        if translations:
            # Try citizen's own language first
            if citizen_language in translations:
                t = translations[citizen_language]
                if t.get("success") and t.get("text"):
                    logger.info(f"Sending in citizen language: {citizen_language}")
                    return t["text"]

            # Try English next
            if "en" in translations:
                t = translations["en"]
                if t.get("success") and t.get("text"):
                    logger.info("Sending in English")
                    return t["text"]

            # Try Hindi as last resort
            if "hi" in translations:
                t = translations["hi"]
                if t.get("success") and t.get("text"):
                    logger.info("Sending in Hindi")
                    return t["text"]

        # Final fallback
        logger.info("Composing English fallback")
        return format_whatsapp_message(
            claim=claim,
            verdict=verdict,
            score=score,
            reasoning=reasoning,
            sources=sources,
        )

    # ── Twilio WhatsApp sender ────────────────────────────────

    async def _send_whatsapp(
        self, phone_number: str, message: str
    ) -> dict:
        if not self._twilio:
            return await self._simulate(phone_number, message)

        try:
            # Format number
            number = phone_number.strip()
            if not number.startswith("+"):
                number = f"+{number}"

            # Run Twilio in thread (it's synchronous)
            loop = asyncio.get_event_loop()
            msg  = await loop.run_in_executor(
                None,
                lambda: self._twilio.messages.create(
                    from_=settings.twilio_whatsapp_from,
                    to=f"whatsapp:{number}",
                    body=message,
                )
            )

            logger.success(
                f"WhatsApp sent to {number} | SID: {msg.sid}"
            )
            return {
                "status":     "sent",
                "channel":    "whatsapp",
                "message_id": msg.sid,
                "recipient":  number,
                "sent_at":    datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Twilio send failed: {e}")
            # Fall back to simulation
            return await self._simulate(phone_number, message)

    # ── Send ACK to citizen ───────────────────────────────────

    async def send_ack(self, phone_number: str):
        """Send immediate acknowledgement to citizen."""
        ack = (
            "✅ *Received.*\n\n"
            "We're checking this with government sources.\n"
            "You'll hear back shortly.\n\n"
            "— TruthSetu"
        )
        await self._send_whatsapp(phone_number, ack)

    # ── Twilio webhook receiver ───────────────────────────────

    async def process_incoming_webhook(
        self, form_data: dict
    ) -> Optional[dict]:
        """
        Process incoming WhatsApp message from Twilio webhook.
        Twilio sends form data (not JSON).

        Webhook URL to set in Twilio console:
        https://your-domain.com/api/v1/webhook/whatsapp
        """
        try:
            text   = form_data.get("Body", "").strip()
            sender = form_data.get("From", "")

            # Remove whatsapp: prefix
            sender_number = sender.replace("whatsapp:", "").strip()

            if not text or not sender_number:
                return None

            logger.info(
                f"Incoming WhatsApp [{sender_number}]: "
                f"{text[:60]}..."
            )

            return {
                "text":          text,
                "sender_number": sender_number,
                "platform":      "whatsapp",
            }

        except Exception as e:
            logger.warning(f"Webhook processing failed: {e}")
            return None

    # ── Simulate (dev mode) ───────────────────────────────────

    async def _simulate(
        self, recipient: str, message: str
    ) -> dict:
        logger.info(
            f"\n{'='*50}\n"
            f"[SIMULATED WHATSAPP] → {recipient}\n"
            f"{'─'*50}\n"
            f"{message}\n"
            f"{'='*50}"
        )
        return {
            "status":    "simulated",
            "channel":   "whatsapp",
            "recipient": recipient,
            "sent_at":   datetime.now(timezone.utc).isoformat(),
        }

    # ── MongoDB log ───────────────────────────────────────────

    async def _log_deployment(
        self,
        verdict_id:    Optional[str],
        sender_number: str,
        platform:      str,
        message:       str,
        language:      str,
        result:        dict,
    ):
        try:
            from backend.db.mongodb import get_db
            db = get_db()
            await db.deployments.insert_one({
                "verdict_id":      verdict_id,
                "sender_number":   sender_number,
                "platform":        platform,
                "language":        language,
                "message_preview": message[:200],
                "status":          result.get("status"),
                "channel":         result.get("channel"),
                "message_id":      result.get("message_id"),
                "error":           result.get("error"),
                "deployed_at":     datetime.now(timezone.utc),
            })
        except Exception as e:
            logger.warning(f"Deployment log failed: {e}")


# ── Singleton ─────────────────────────────────────────────────
_agent: Optional[DeployAgent] = None


async def get_deploy_agent() -> DeployAgent:
    global _agent
    if _agent is None:
        _agent = DeployAgent()
        await _agent.initialize()
    return _agent