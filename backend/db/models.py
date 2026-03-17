from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class Verdict(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNVERIFIABLE = "UNVERIFIABLE"
    PENDING = "PENDING"


class ClaimType(str, Enum):
    EPHEMERAL = "EPHEMERAL"
    SEMI_PERMANENT = "SEMI_PERMANENT"
    PERMANENT = "PERMANENT"


class Platform(str, Enum):
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    NEWS_API = "news_api"
    MANUAL = "manual"


class ClaimDocument(BaseModel):
    claim_text: str
    platform: Platform = Platform.MANUAL
    source_url: Optional[str] = None
    sender_number: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    language: str = "en"
    crisis_type: Optional[str] = None
    claim_type: Optional[str] = None
    is_duplicate: bool = False
    status: str = "received"


class VerdictDocument(BaseModel):
    claim: str
    verdict: Verdict = Verdict.PENDING
    credibility_score: int = 0
    reasoning: str = ""
    sources: List[str] = []
    claim_type: str = "EPHEMERAL"
    platform: str = "manual"
    verified_at: Optional[datetime] = None
    is_overridden: bool = False
    override_reason: Optional[str] = None
    override_by: Optional[str] = None
    from_cache: bool = False


class TranslationDocument(BaseModel):
    verdict_id: str
    correction_text_en: str
    translations: dict = {}
    translated_at: datetime = Field(default_factory=datetime.utcnow)


class DeploymentDocument(BaseModel):
    verdict_id: str
    sender_number: Optional[str] = None
    channels: List[str] = []
    languages_sent: List[str] = []
    recipients_count: int = 0
    deployed_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "sent"


class TemplateDocument(BaseModel):
    template_text: str
    example_claim: str
    crisis_type: str
    claim_type: str
    verdict: str
    credibility_score: int
    source: str
    explanation: str
    embedding: Optional[List[float]] = None
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    occurrence_count: int = 1