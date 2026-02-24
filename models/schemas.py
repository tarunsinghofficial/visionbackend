"""Pydantic models for Vision-Sync API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── CV Service Models ──────────────────────────────────────────────

class DetectedObject(BaseModel):
    """A single object detected by YOLOv8."""

    label: str
    confidence: float = Field(..., ge=0, le=1)
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] pixel coords")


# ── Vector Service Models ──────────────────────────────────────────

class VectorMatch(BaseModel):
    """A furniture product returned from ChromaDB."""

    id: str
    name: str
    description: str
    category: str
    style: str
    room_type: str
    similarity_score: float = Field(default=0.0)


# ── GenAI Service Models ──────────────────────────────────────────

class ImprovementSuggestion(BaseModel):
    area: str
    suggestion: str
    priority: str = Field(..., pattern="^(high|medium|low)$")


class GeminiAnalysis(BaseModel):
    room_type: str = "unknown"
    room_summary: str = ""
    style_detected: str = "unknown"
    improvement_suggestions: list[ImprovementSuggestion] = []
    color_palette_recommendation: list[str] = []
    estimated_improvement_score: float = Field(default=5.0, ge=1, le=10)
    furniture_to_add: list[str] = []
    furniture_to_remove: list[str] = []


# ── API Response Models ───────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """Full response returned by POST /api/analyze."""

    detected_objects: list[DetectedObject]
    vector_recommendations: list[VectorMatch]
    analysis: GeminiAnalysis
    annotated_image: str = Field(..., description="Base64-encoded annotated image")
    image_url: Optional[str] = None


class HistoryItem(BaseModel):
    """A single past analysis record."""

    id: str
    image_url: Optional[str] = None
    room_type: str
    style_detected: str
    improvement_score: float
    detected_objects: list[DetectedObject] = []
    full_analysis: Optional[GeminiAnalysis] = None
    created_at: datetime
