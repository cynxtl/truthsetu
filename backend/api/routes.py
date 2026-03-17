from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.agents.verify_agent import get_verify_agent, VerifyAgent
from backend.agents.scout_agent import get_scout_agent, ScoutAgent
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
    )
    return result


@router.get("/scout/status", tags=["SCOUT"])
async def scout_status(scout: ScoutAgent = Depends(get_scout_agent)):
    return await scout.get_status()


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
            "languages_active":     5,
            "avg_response_minutes": 8.3,
        }
    except Exception:
        return {
            "claims_today": 0, "false_claims_caught": 0,
            "corrections_deployed": 0, "languages_active": 0,
            "avg_response_minutes": 0,
        }

# ── PIPELINE ──────────────────────────────────────────────────
class PipelineRequest(BaseModel):
    text: str
    platform: str = "manual"
    sender_number: Optional[str] = None
    language: str = "en"
    metrics: Optional[dict] = None


@router.post("/pipeline/run", tags=["Pipeline"])
async def run_pipeline(
    payload: PipelineRequest,
    pipeline: TruthSetuPipeline = Depends(get_pipeline)
):
    """Run the full pipeline: SCOUT → VERIFY → TRANSLATE → DEPLOY → LEARN"""
    result = await pipeline.run(
        text=payload.text,
        platform=payload.platform,
        sender_number=payload.sender_number,
        language=payload.language,
        metrics=payload.metrics or {},
    )
    return result