import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from backend.agents.verify_agent import get_verify_agent, VerifyAgent
from backend.agents.scout_agent import get_scout_agent, ScoutAgent
from backend.agents.translate_agent import get_translate_agent, TranslateAgent
from backend.db.mongodb import get_db
from backend.core.pipeline import get_pipeline, TruthSetuPipeline

router = APIRouter()


# ── Request schemas ───────────────────────────────────────────
class ClaimRequest(BaseModel):
    claim: str
    platform: str = "manual"
    language: str = "en"


class OverrideRequest(BaseModel):
    verdict: str
    reason: str
    admin_id: str


class SubmitRequest(BaseModel):
    text: str
    platform: str = "manual"
    sender_number: Optional[str] = None
    source_url: Optional[str] = None
    metrics: Optional[dict] = None


class TranslateRequest(BaseModel):
    message: str
    target_languages: Optional[list[str]] = None
    citizen_language: Optional[str] = None


class PipelineRequest(BaseModel):
    text: str
    platform: str = "manual"
    sender_number: Optional[str] = None
    language: str = "en"
    metrics: Optional[dict] = None


# ── VERIFY ────────────────────────────────────────────────────
@router.post("/verify", tags=["VERIFY"])
async def verify_claim(
    payload: ClaimRequest,
    agent: VerifyAgent = Depends(get_verify_agent)
):
    if not payload.claim.strip():
        raise HTTPException(status_code=422, detail="Claim cannot be empty.")
    result = await agent.verify(payload.claim, payload.platform)
    try:
        db = get_db()
        await db.verdicts.insert_one({
            **{k: v for k, v in result.items() if k != "retrieved_docs"},
        })
    except Exception:
        pass
    return result


@router.post("/verify/rebuild-index", tags=["VERIFY"])
async def rebuild_index(agent: VerifyAgent = Depends(get_verify_agent)):
    await agent.rebuild_index()
    return {"status": "Index rebuilt successfully"}


# ── SCOUT ─────────────────────────────────────────────────────
@router.post("/scout/submit", tags=["SCOUT"])
async def submit_claim(
    payload: SubmitRequest,
    scout: ScoutAgent = Depends(get_scout_agent)
):
    result = await scout.process(
        text=payload.text,
        platform=payload.platform,
        sender_number=payload.sender_number,
        source_url=payload.source_url,
        metrics=payload.metrics or {},
    )
    return result


@router.get("/scout/status", tags=["SCOUT"])
async def scout_status(scout: ScoutAgent = Depends(get_scout_agent)):
    return await scout.get_status()


# ── TRANSLATE ─────────────────────────────────────────────────
@router.post("/translate", tags=["TRANSLATE"])
async def translate_message(
    payload: TranslateRequest,
    agent: TranslateAgent = Depends(get_translate_agent)
):
    if not payload.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    result = await agent.translate(
        message=payload.message,
        target_languages=payload.target_languages,
        citizen_language=payload.citizen_language,
    )
    return result


@router.post("/translate/detect-language", tags=["TRANSLATE"])
async def detect_language(
    payload: dict,
    agent: TranslateAgent = Depends(get_translate_agent)
):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=422, detail="Text required.")
    lang = await agent.detect_language(text)
    return {"language": lang, "text": text[:50]}


# ── PIPELINE ──────────────────────────────────────────────────
@router.post("/pipeline/run", tags=["Pipeline"])
async def run_pipeline(
    payload: PipelineRequest,
    pipeline: TruthSetuPipeline = Depends(get_pipeline)
):
    result = await pipeline.run(
        text=payload.text,
        platform=payload.platform,
        sender_number=payload.sender_number,
        language=payload.language,
        metrics=payload.metrics or {},
    )
    return result


# ── WHATSAPP WEBHOOK ──────────────────────────────────────────
@router.post("/webhook/whatsapp", tags=["WhatsApp"])
async def whatsapp_webhook(request: Request):
    """
    Twilio webhook — receives incoming WhatsApp messages.
    Set this URL in Twilio console under sandbox settings.
    """
    from backend.agents.deploy_agent import get_deploy_agent

    # Twilio sends form data not JSON
    form_data = await request.form()
    form_dict = dict(form_data)

    deploy   = await get_deploy_agent()
    incoming = await deploy.process_incoming_webhook(form_dict)

    if not incoming:
        return Response(
            content="<Response></Response>",
            media_type="application/xml"
        )

    # Send ACK immediately
    await deploy.send_ack(incoming["sender_number"])

    # Run pipeline in background
    pipeline = await get_pipeline()
    asyncio.create_task(
        pipeline.run(
            text=incoming["text"],
            platform="whatsapp",
            sender_number=incoming["sender_number"],
        )
    )

    return Response(
        content="<Response></Response>",
        media_type="application/xml"
    )


# ── DEPLOY STATUS ─────────────────────────────────────────────
@router.get("/deploy/status", tags=["DEPLOY"])
async def deploy_status():
    try:
        db        = get_db()
        total     = await db.deployments.count_documents({})
        sent      = await db.deployments.count_documents({"status": "sent"})
        failed    = await db.deployments.count_documents({"status": "failed"})
        simulated = await db.deployments.count_documents({"status": "simulated"})
        return {
            "total_deployments": total,
            "sent":              sent,
            "failed":            failed,
            "simulated":         simulated,
        }
    except Exception as e:
        return {"error": str(e)}


# ── DASHBOARD ─────────────────────────────────────────────────
@router.get("/claims", tags=["Dashboard"])
async def get_claims(limit: int = 20, skip: int = 0):
    try:
        db     = get_db()
        cursor = db.verdicts.find().sort("verified_at", -1).skip(skip).limit(limit)
        claims = await cursor.to_list(length=limit)
        for c in claims:
            c["_id"] = str(c["_id"])
        total = await db.verdicts.count_documents({})
        return {"claims": claims, "total": total}
    except Exception as e:
        return {"claims": [], "total": 0, "error": str(e)}


@router.post("/claims/{claim_id}/override", tags=["Dashboard"])
async def override_verdict(claim_id: str, payload: OverrideRequest):
    try:
        from bson import ObjectId
        db = get_db()
        await db.verdicts.update_one(
            {"_id": ObjectId(claim_id)},
            {"$set": {
                "verdict":         payload.verdict,
                "is_overridden":   True,
                "override_reason": payload.reason,
                "override_by":     payload.admin_id,
            }}
        )
        return {"status": "override recorded", "claim_id": claim_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stats", tags=["Dashboard"])
async def get_stats():
    try:
        db          = get_db()
        total       = await db.verdicts.count_documents({})
        false_count = await db.verdicts.count_documents({"verdict": "FALSE"})
        deployed    = await db.deployments.count_documents({})
        return {
            "claims_today":         total,
            "false_claims_caught":  false_count,
            "corrections_deployed": deployed,
            "languages_active":     2,
            "avg_response_minutes": 2.0,
        }
    except Exception:
        return {
            "claims_today": 0, "false_claims_caught": 0,
            "corrections_deployed": 0, "languages_active": 0,
            "avg_response_minutes": 0,
        }