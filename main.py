"""Vision-Sync FastAPI Backend — main entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analyze, history

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown."""
    logger.info("🚀 Vision-Sync backend starting …")
    yield
    logger.info("👋 Vision-Sync backend shutting down.")


app = FastAPI(
    title="Vision-Sync API",
    description="Intelligent Space Analyzer — CV + GenAI room analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────
# Allow Vercel deployment URLs via a callback
_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]


def _cors_origin_callback(origin: str) -> bool:
    """Allow explicit origins + any *.vercel.app subdomain."""
    if origin in _ALLOWED_ORIGINS:
        return True
    if origin.endswith(".vercel.app"):
        return True
    return False


app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────
app.include_router(analyze.router)
app.include_router(history.router)


# ── Health Check ──────────────────────────────────────────────────
@app.get("/api/health", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "vision-sync"}
