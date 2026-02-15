from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import crud
from ..models.datapoint import DataPointCreate, DataPointResponse, ChartDataRequest, ChartDataResponse, SystemStats
from ..db.session import get_db
from ..services.analytics import AnalyticsService

router = APIRouter(prefix="/datapoints", tags=["datapoints"])

analytics_service = AnalyticsService()


@router.post("/", response_model=DataPointResponse)
def create_datapoint(datapoint: DataPointCreate, db: Session = Depends(get_db)):
    """Create a new data point"""
    try:
        return crud.create_datapoint(db, datapoint)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{datapoint_id}", response_model=DataPointResponse)
def get_datapoint(datapoint_id: int, db: Session = Depends(get_db)):
    """Get a specific data point by ID"""
    datapoint = crud.get_datapoint(db, datapoint_id)
    if not datapoint:
        raise HTTPException(status_code=404, detail="Data point not found")
    return datapoint


@router.get("/", response_model=List[DataPointResponse])
def list_datapoints(
    skip: int = 0,
    limit: int = 100,
    source: str = None,
    metric_type: str = None,
    db: Session = Depends(get_db)
):
    """List data points with optional filtering"""
    return crud.get_datapoints(db, skip=skip, limit=limit, source=source, metric_type=metric_type)


@router.post("/chart-data", response_model=ChartDataResponse)
def get_chart_data(request: ChartDataRequest, db: Session = Depends(get_db)):
    """Get chart data for visualization"""
    try:
        return analytics_service.get_chart_data(db, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stats", response_model=SystemStats)
def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    return analytics_service.get_system_stats(db)