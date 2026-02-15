from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text
from ..db.session import Base


class DataPointType(Enum):
    METRIC = "metric"
    EVENT = "event"
    LOG = "log"


class DataPoint(Base):
    __tablename__ = "data_points"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    source = Column(String(255), nullable=False)
    metric_type = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    value_str = Column(Text)  # For non-numeric values
    tags = Column(Text)  # JSON string for tags
    is_active = Column(Boolean, default=True)
    

class DataPointCreate(BaseModel):
    timestamp: Optional[datetime] = None
    source: str
    metric_type: str
    value: float
    value_str: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None


class DataPointResponse(DataPointCreate):
    id: int
    timestamp: datetime
    is_active: bool = True


class ChartDataRequest(BaseModel):
    metric_types: List[str]
    start_time: datetime
    end_time: datetime
    group_by: Optional[str] = "hour"  # hour, day, week, month
    limit: Optional[int] = 100
    filters: Optional[Dict[str, Any]] = None


class ChartDataResponse(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SystemStats(BaseModel):
    total_data_points: int
    active_sources: int
    avg_processing_time: float
    success_rate: float
    memory_usage: float
    cpu_usage: float