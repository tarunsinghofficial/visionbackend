"""GenAI service — Google Gemini 2.0 Flash for room analysis."""

from __future__ import annotations

import json
import logging
from typing import Optional

import google.generativeai as genai

from models.schemas import GeminiAnalysis, DetectedObject, VectorMatch

logger = logging.getLogger(__name__)

_configured = False


def _ensure_configured(api_key: str) -> None:
    global _configured
    if not _configured:
        genai.configure(api_key=api_key)
        _configured = True


def _infer_room_type(objects: list[str]) -> str:
    """Heuristic room-type inference from detected objects."""
    obj_set = set(objects)
    if obj_set & {"bed"}:
        return "bedroom"
    if obj_set & {"toilet"}:
        return "bathroom"
    if obj_set & {"refrigerator", "oven", "microwave"}:
        return "kitchen"
    if obj_set & {"sink"} and not obj_set & {"toilet"}:
        # Sink without toilet is far more likely a kitchen
        return "kitchen"
    if obj_set & {"couch", "tv"}:
        return "living room"
    if obj_set & {"dining table"}:
        return "dining room"
    if obj_set & {"laptop", "chair"}:
        return "office"
    return "living room"


def _build_prompt(
    detected_objects: list[DetectedObject],
    room_type: str,
    recommendations: list[VectorMatch],
) -> str:
    obj_list = [f"- {o.label} (confidence {o.confidence:.0%})" for o in detected_objects]
    rec_list = [
        f"- {r.name}: {r.description} (style: {r.style})"
        for r in recommendations
    ]

    return f"""You are an expert interior designer and space analyst. Analyze the following room data and provide improvement suggestions.

DETECTED OBJECTS:
{chr(10).join(obj_list) if obj_list else "- No furniture-relevant objects detected"}

INFERRED ROOM TYPE: {room_type}

RECOMMENDED PRODUCTS FROM OUR CATALOG:
{chr(10).join(rec_list) if rec_list else "- No recommendations available"}

Based on this analysis, respond ONLY with a valid JSON object (no markdown, no code fences, no extra text). The JSON must have exactly this structure:
{{
  "room_type": "{room_type}",
  "room_summary": "2-3 sentences describing the current state of the room",
  "style_detected": "one of: minimalist, modern, traditional, industrial, bohemian, scandinavian, cluttered, sparse, eclectic",
  "improvement_suggestions": [
    {{ "area": "specific area", "suggestion": "actionable suggestion", "priority": "high|medium|low" }}
  ],
  "color_palette_recommendation": ["#hex1", "#hex2", "#hex3"],
  "estimated_improvement_score": <number 1-10>,
  "furniture_to_add": ["item1", "item2"],
  "furniture_to_remove": ["item1"]
}}

Respond ONLY in valid JSON. No markdown formatting, no code blocks, no explanations."""


# Models to try in order — if one hits quota, try the next
_MODEL_CHAIN = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]


def _parse_gemini_response(text: str) -> dict:
    """Strip markdown fences and parse JSON from Gemini response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return json.loads(text)


async def generate_analysis(
    detected_objects: list[DetectedObject],
    recommendations: list[VectorMatch],
    api_key: str,
) -> GeminiAnalysis:
    """Call Gemini to generate a room analysis, trying multiple models.

    Tries each model in _MODEL_CHAIN. If one fails (e.g. quota exceeded),
    falls back to the next. If all fail, returns a basic CV-based analysis.
    """
    labels = [o.label for o in detected_objects]
    room_type = _infer_room_type(labels)

    if not api_key:
        logger.warning("No Gemini API key — returning fallback analysis.")
        return _fallback_analysis(detected_objects, recommendations, room_type)

    _ensure_configured(api_key)
    prompt = _build_prompt(detected_objects, room_type, recommendations)

    for model_name in _MODEL_CHAIN:
        try:
            logger.info("Trying Gemini model: %s", model_name)
            model = genai.GenerativeModel(model_name)
            response = await model.generate_content_async(prompt)

            data = _parse_gemini_response(response.text)
            analysis = GeminiAnalysis(**data)
            logger.info("Gemini analysis completed with model: %s", model_name)
            return analysis

        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "quota" in error_str.lower():
                logger.warning("Model %s quota exceeded, trying next…", model_name)
                continue
            else:
                logger.error("Gemini model %s failed: %s", model_name, exc)
                break  # Non-quota error, don't bother trying other models

    logger.warning("All Gemini models failed — using fallback analysis.")
    return _fallback_analysis(detected_objects, recommendations, room_type)


def _fallback_analysis(
    detected_objects: list[DetectedObject],
    recommendations: list[VectorMatch],
    room_type: str,
) -> GeminiAnalysis:
    """Build a basic analysis from CV + vector data when Gemini is unavailable."""
    labels = [o.label for o in detected_objects]
    unique_labels = list(set(labels))

    suggestions = []
    if len(unique_labels) < 3:
        suggestions.append({
            "area": "General",
            "suggestion": "The room appears sparse. Consider adding more furniture for a complete look.",
            "priority": "medium",
        })
    if "potted plant" not in labels:
        suggestions.append({
            "area": "Greenery",
            "suggestion": "Add indoor plants to bring life and color to the space.",
            "priority": "low",
        })

    furniture_to_add = [r.name for r in recommendations[:3]]

    return GeminiAnalysis(
        room_type=room_type,
        room_summary=f"A {room_type} containing {', '.join(unique_labels) if unique_labels else 'no detected furniture'}. "
                      f"Analysis generated from computer vision results (AI summary unavailable).",
        style_detected="undetermined",
        improvement_suggestions=[
            GeminiAnalysis.model_fields["improvement_suggestions"].annotation.__args__[0](**s)
            for s in suggestions
        ] if suggestions else [],
        color_palette_recommendation=["#3b82f6", "#1e293b", "#f8fafc"],
        estimated_improvement_score=5.0,
        furniture_to_add=furniture_to_add,
        furniture_to_remove=[],
    )
