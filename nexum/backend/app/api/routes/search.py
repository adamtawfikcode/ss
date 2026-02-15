"""
Nexum API — Search routes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import SearchFilters, SearchRequest, SearchResponse
from app.services.search.search_service import search_service

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest, db: AsyncSession = Depends(get_db)):
    """
    Multimodal semantic search across video moments.

    Accepts natural language queries like:
    - "guy struggling playing tetris at score 1296"
    - "podcast where they argue about AI ethics"
    """
    return await search_service.search(request, db)


@router.get("", response_model=SearchResponse)
async def search_get(
    q: str = Query(..., min_length=1, max_length=1024),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    language: str | None = None,
    min_confidence: float | None = Query(None, ge=0, le=1),
    min_duration: int | None = None,
    max_duration: int | None = None,
    min_views: int | None = None,
    modality: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """GET-based search endpoint for simpler integrations."""
    filters = SearchFilters(
        language=language,
        min_confidence=min_confidence,
        min_duration=min_duration,
        max_duration=max_duration,
        min_views=min_views,
        modality=modality,
    )
    request = SearchRequest(query=q, filters=filters, page=page, page_size=page_size)
    return await search_service.search(request, db)
