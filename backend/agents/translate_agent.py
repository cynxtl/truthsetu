"""
TruthSetu — TRANSLATE Agent
============================
Translates verified corrections into Indian languages.

Current backend: Groq (Llama-3.1-8b-instant)
Future backend:  IndicTrans2 (swap _translate_with_groq 
                 with _translate_with_indictrans2)

Supported languages at launch:
  hi → Hindi
  mr → Marathi

Adding more languages later:
  Just add to SUPPORTED_LANGUAGES dict — nothing else changes.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

from backend.core.config import get_settings

settings = get_settings()

# ── Supported languages ───────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "hi": {
        "name":        "Hindi",
        "native_name": "हिन्दी",
        "script":      "Devanagari",
    },
    "mr": {
        "name":        "Marathi",
        "native_name": "मराठी",
        "script":      "Devanagari",
    },
}

# ── Translation prompts per language ─────────────────────────
TRANSLATION_PROMPTS = {
    "hi": """You are a professional Hindi translator for India's 
crisis fact-checking system TruthSetu.

Translate this WhatsApp fact-check message to Hindi.

Rules:
- Keep emojis exactly as they are (🔴 🟢 🟡 ❌ ✅ ⚠️ 📎 📊 ⏱)
- Keep URLs exactly as they are
- Keep "TruthSetu" as is (do not translate the name)
- Use simple, clear Hindi that anyone can understand
- Do NOT use overly formal or bureaucratic language
- Crisis terms should be clear and direct
- Keep the same message structure and line breaks

Original message:
{message}

Translate to Hindi (respond with translation only, nothing else):""",

    "mr": """You are a professional Marathi translator for India's 
crisis fact-checking system TruthSetu.

Translate this WhatsApp fact-check message to Marathi.

Rules:
- Keep emojis exactly as they are (🔴 🟢 🟡 ❌ ✅ ⚠️ 📎 📊 ⏱)
- Keep URLs exactly as they are
- Keep "TruthSetu" as is (do not translate the name)
- Use simple, clear Marathi that anyone can understand
- Use standard Maharashtra Marathi (not regional dialect)
- Crisis terms should be clear and direct
- Keep the same message structure and line breaks

Original message:
{message}

Translate to Marathi (respond with translation only, nothing else):""",
}

# ── Language detection prompt ─────────────────────────────────
DETECT_LANGUAGE_PROMPT = """Detect the language of this text.
Respond with ONLY the 2-letter language code.

Supported codes:
  en = English
  hi = Hindi
  mr = Marathi
  ta = Tamil
  te = Telugu
  bn = Bengali
  kn = Kannada
  gu = Gujarati
  ml = Malayalam
  pa = Punjabi
  ur = Urdu

Text: "{text}"

Respond with only the 2-letter code:"""


class TranslateAgent:

    def __init__(self):
        self._groq  = None
        self._ready = False

    async def initialize(self):
        from groq import Groq
        logger.info("Initialising TRANSLATE agent...")
        self._groq  = Groq(api_key=settings.groq_api_key)
        self._ready = True
        logger.success("TRANSLATE agent ready ✓")

    # ── Language detection ────────────────────────────────────

    async def detect_language(self, text: str) -> str:
        """
        Detect the language of incoming text.
        Returns 2-letter language code (en, hi, mr, ta etc.)
        """
        if not self._ready:
            await self.initialize()

        try:
            response = self._groq.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {
                        "role":    "user",
                        "content": DETECT_LANGUAGE_PROMPT.format(
                            text=text[:200]
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=5,
            )
            lang = response.choices[0].message.content.strip().lower()

            # Validate — must be a known code
            valid_codes = [
                "en", "hi", "mr", "ta", "te",
                "bn", "kn", "gu", "ml", "pa", "ur"
            ]
            if lang in valid_codes:
                logger.info(f"Detected language: {lang}")
                return lang

            return "en"  # default to English

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"

    # ── Single language translation ───────────────────────────

    async def translate_to(
        self, message: str, target_lang: str
    ) -> dict:
        """
        Translate a message to a single target language.

        Returns:
          {
            "language":     "hi",
            "language_name":"Hindi",
            "text":         "translated text",
            "success":      True/False,
            "fallback":     False  (True if we fell back to English)
          }
        """
        if not self._ready:
            await self.initialize()

        if target_lang not in SUPPORTED_LANGUAGES:
            return {
                "language":      target_lang,
                "language_name": target_lang,
                "text":          message,
                "success":       False,
                "fallback":      True,
                "error":         f"Language '{target_lang}' not supported yet",
            }

        lang_info = SUPPORTED_LANGUAGES[target_lang]

        try:
            translated = await self._translate_with_groq(
                message, target_lang
            )

            # Validate translation isn't empty or just returned English
            if not translated or len(translated) < 10:
                raise ValueError("Translation too short")

            logger.info(
                f"Translated to {lang_info['name']}: "
                f"{translated[:60]}..."
            )

            return {
                "language":      target_lang,
                "language_name": lang_info["name"],
                "native_name":   lang_info["native_name"],
                "text":          translated,
                "success":       True,
                "fallback":      False,
            }

        except Exception as e:
            logger.warning(
                f"Translation to {lang_info['name']} failed: {e} "
                f"— falling back to English"
            )
            return {
                "language":      target_lang,
                "language_name": lang_info["name"],
                "native_name":   lang_info["native_name"],
                "text":          message,   # fallback to English
                "success":       False,
                "fallback":      True,
                "error":         str(e),
            }

    # ── Multi-language translation ────────────────────────────

    async def translate(
        self,
        message: str,
        target_languages: Optional[list[str]] = None,
        citizen_language: Optional[str] = None,
    ) -> dict:
        """
        Main entry point.
        Translates message to all target languages concurrently.

        Args:
          message:          English correction text
          target_languages: list of language codes to translate to
                            defaults to all SUPPORTED_LANGUAGES
          citizen_language: if provided, this language is prioritised
                            (citizen's detected language)

        Returns:
          {
            "original":      "English message",
            "translations":  {
              "hi": { "text": "...", "success": True, ... },
              "mr": { "text": "...", "success": True, ... },
            },
            "priority_language": "hi",
            "translated_at": "ISO timestamp",
          }
        """
        if not self._ready:
            await self.initialize()

        if target_languages is None:
            target_languages = list(SUPPORTED_LANGUAGES.keys())

        logger.info(
            f"Translating to: {', '.join(target_languages)}"
        )

        # Translate all languages concurrently
        tasks = [
            self.translate_to(message, lang)
            for lang in target_languages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        translations = {}
        for lang, result in zip(target_languages, results):
            if isinstance(result, Exception):
                translations[lang] = {
                    "language":      lang,
                    "language_name": SUPPORTED_LANGUAGES.get(
                        lang, {}
                    ).get("name", lang),
                    "text":          message,
                    "success":       False,
                    "fallback":      True,
                    "error":         str(result),
                }
            else:
                translations[lang] = result

        # Determine priority language
        # If citizen sent message in Hindi → send Hindi reply first
        priority = citizen_language \
            if citizen_language in translations \
            else (target_languages[0] if target_languages else "en")

        successful = sum(
            1 for t in translations.values() if t.get("success")
        )
        logger.success(
            f"Translation complete: "
            f"{successful}/{len(target_languages)} successful"
        )

        return {
            "original":          message,
            "translations":      translations,
            "priority_language": priority,
            "languages_count":   len(translations),
            "success_count":     successful,
            "translated_at":     datetime.now(timezone.utc).isoformat(),
        }

    # ── Groq translation backend ──────────────────────────────

    async def _translate_with_groq(
        self, message: str, target_lang: str
    ) -> str:
        """
        Translate using Groq Llama-3.1.
        Can be swapped for IndicTrans2 later without
        changing anything else.
        """
        prompt = TRANSLATION_PROMPTS.get(target_lang)
        if not prompt:
            raise ValueError(
                f"No translation prompt for language: {target_lang}"
            )

        response = self._groq.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role":    "user",
                    "content": prompt.format(message=message),
                }
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        return response.choices[0].message.content.strip()

    # ── Future: IndicTrans2 backend ───────────────────────────
    # Uncomment and install IndicTrans2 when ready for production
    #
    # async def _translate_with_indictrans2(
    #     self, message: str, target_lang: str
    # ) -> str:
    #     from IndicTransToolkit import IndicProcessor
    #     from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    #     ...

    # ── Save to MongoDB ───────────────────────────────────────

    async def save_translation(
        self,
        verdict_id: str,
        correction_en: str,
        translation_result: dict,
    ):
        """Save translation result to MongoDB."""
        try:
            from backend.db.mongodb import get_db
            db = get_db()
            await db.translations.insert_one({
                "verdict_id":       verdict_id,
                "correction_text_en": correction_en,
                "translations":     translation_result["translations"],
                "priority_language": translation_result["priority_language"],
                "translated_at":    datetime.now(timezone.utc),
            })
        except Exception as e:
            logger.warning(f"Failed to save translation: {e}")


# ── Singleton ─────────────────────────────────────────────────
_agent: Optional[TranslateAgent] = None


async def get_translate_agent() -> TranslateAgent:
    global _agent
    if _agent is None:
        _agent = TranslateAgent()
        await _agent.initialize()
    return _agent