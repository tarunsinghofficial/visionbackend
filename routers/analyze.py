"""Router for room image analysis."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.schemas import AnalysisResponse
from services import cv_service, genai_service, vector_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])


def _get_supabase():
    """Lazy Supabase client import to avoid errors when env vars are missing."""
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_KEY", "")
        if url and key:
            return create_client(url, key)
    except Exception as exc:
        logger.warning("Supabase not available: %s", exc)
    return None


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    user_id: str | None = Form(default=None),
):
    """Analyze a room image through the full CV → Vector → GenAI pipeline."""

    # ── Validate upload ───────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    # ── Step 1: Computer Vision ───────────────────────────────────
    try:
        model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
        detected_objects, annotated_b64 = cv_service.analyze_image(
            image_bytes, model_path=model_path
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("CV service error: %s", exc)
        raise HTTPException(status_code=500, detail="Image analysis failed.")

    # ── Step 2: Vector Search ─────────────────────────────────────
    labels = list({o.label for o in detected_objects})
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_store")
    recommendations = vector_service.query_recommendations(labels, chroma_path)

    # ── Step 3: GenAI Analysis ────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY", "")
    analysis = await genai_service.generate_analysis(
        detected_objects, recommendations, api_key
    )

    # ── Step 4: Save to Supabase ──────────────────────────────────
    image_url: str | None = None
    supabase = _get_supabase()
    if supabase:
        try:
            # Upload image to Supabase Storage
            file_name = f"{uuid.uuid4()}.jpg"
            storage_path = f"room-images/{file_name}"
            supabase.storage.from_("room-images").upload(
                storage_path,
                image_bytes,
                file_options={"content-type": file.content_type or "image/jpeg"},
            )
            image_url = supabase.storage.from_("room-images").get_public_url(storage_path)

            # Insert analysis record
            record = {
                "user_id": user_id,
                "image_url": image_url,
                "detected_objects": [o.model_dump() for o in detected_objects],
                "room_type": analysis.room_type,
                "style_detected": analysis.style_detected,
                "improvement_score": analysis.estimated_improvement_score,
                "full_analysis": analysis.model_dump(),
            }
            supabase.table("analyses").insert(record).execute()
            logger.info("Analysis saved to Supabase.")
        except Exception as exc:
            logger.error("Supabase save failed (non-blocking): %s", exc)

    return AnalysisResponse(
        detected_objects=detected_objects,
        vector_recommendations=recommendations,
        analysis=analysis,
        annotated_image=annotated_b64,
        image_url=image_url,
    )
