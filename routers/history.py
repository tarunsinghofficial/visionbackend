"""Router for analysis history."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException

from models.schemas import HistoryItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["history"])


def _get_supabase():
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_KEY", "")
        if url and key:
            return create_client(url, key)
    except Exception as exc:
        logger.warning("Supabase not available: %s", exc)
    return None


@router.get("/history/{user_id}", response_model=list[HistoryItem])
async def get_history(user_id: str):
    """Fetch the last 10 analyses for a given user."""

    supabase = _get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.",
        )

    try:
        response = (
            supabase.table("analyses")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )

        items: list[HistoryItem] = []
        for row in response.data:
            items.append(
                HistoryItem(
                    id=row["id"],
                    image_url=row.get("image_url"),
                    room_type=row.get("room_type", "unknown"),
                    style_detected=row.get("style_detected", "unknown"),
                    improvement_score=row.get("improvement_score", 0),
                    detected_objects=row.get("detected_objects", []),
                    full_analysis=row.get("full_analysis"),
                    created_at=row["created_at"],
                )
            )
        return items

    except Exception as exc:
        logger.error("Failed to fetch history: %s", exc)
        raise HTTPException(status_code=500, detail="Could not retrieve analysis history.")
