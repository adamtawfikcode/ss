"""
Nexum — Main FastAPI Application

Multimodal Video Memory Search Engine
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.core.config import get_settings

settings = get_settings()

# ── Logging ──────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(settings.log_level)
    ),
)

logger = structlog.get_logger()


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Starting Nexum", version=settings.app_version, device=settings.device)

    # Initialize database tables
    await init_db()

    # Initialize vector store collections (text + visual + comments)
    from app.services.search.vector_store import vector_store
    vector_store.ensure_collections()

    # Initialize Neo4j graph database
    from app.core.graph_database import graph_db
    try:
        await graph_db.connect()
        await graph_db.ensure_schema()
        logger.info("Neo4j graph database initialized")
    except Exception as e:
        logger.warning(f"Neo4j init failed (graph features degraded): {e}")

    # v4: Load confidence calibrators from disk
    from app.ml.calibration import calibration_service
    try:
        calibration_service.load()
        logger.info("Calibration service initialized", version=calibration_service.version)
    except Exception as e:
        logger.info(f"No calibration data yet (will train on first feedback): {e}")

    logger.info(
        "Nexum ready",
        device=settings.device,
        whisper_model=settings.whisper_model,
        clip_model=settings.clip_model,
        text_model=settings.text_embedding_model,
    )

    yield

    # Shutdown
    try:
        await graph_db.close()
    except Exception:
        pass
    logger.info("Shutting down Nexum")


from app.core.database import init_db

# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nexum",
    description="Calibrated Acoustic Intelligence Engine with Multimodal Social Knowledge Graph",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Routes ───────────────────────────────────────────────────────────────

from app.api.routes import search, videos, admin, feedback, graph, comments, websocket, audio, legal
from app.api.routes import v4 as v4_routes
from app.api.routes import demo as demo_routes
from app.api.routes import queue as queue_routes
from app.api.routes.recrawl_signals import recrawl_routes, signal_routes
from app.api.routes import intelligence as intelligence_routes

app.include_router(search.router, prefix=settings.api_prefix)
app.include_router(videos.router, prefix=settings.api_prefix)
app.include_router(admin.router, prefix=settings.api_prefix)
app.include_router(feedback.router, prefix=settings.api_prefix)
app.include_router(graph.router, prefix=settings.api_prefix)
app.include_router(comments.router, prefix=settings.api_prefix)
app.include_router(websocket.router, prefix=settings.api_prefix)
app.include_router(audio.router, prefix=settings.api_prefix)
app.include_router(legal.router, prefix=settings.api_prefix)
app.include_router(v4_routes.router, prefix=settings.api_prefix)
app.include_router(demo_routes.router, prefix=settings.api_prefix)
app.include_router(queue_routes.router, prefix=settings.api_prefix)
app.include_router(recrawl_routes, prefix=settings.api_prefix)
app.include_router(signal_routes, prefix=settings.api_prefix)
app.include_router(intelligence_routes.router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {
        "name": "Nexum",
        "description": "Calibrated Acoustic Intelligence Engine",
        "version": settings.app_version,
        "device": settings.device,
        "modalities": 15,
        "features": [
            "demo_mode", "priority_queue", "graph_visualization",
            "deep_signals", "recrawl_tracking", "staleness_detection",
        ],
        "signal_categories": [
            "temporal_visual", "audio_micro", "behavioral",
            "comment_archaeology", "cross_video_forensics",
            "linguistic", "graph_relational",
        ],
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "device": settings.device}
