from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import random
from datetime import datetime, timedelta

app = FastAPI(
    title="Nexum API",
    description="Advanced analytics and data visualization backend",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class DataPoint(BaseModel):
    timestamp: datetime
    value: float
    source: str

class ChartDataRequest(BaseModel):
    metric: str
    timeframe: str  # '1d', '7d', '30d', '90d'
    filters: Optional[Dict[str, Any]] = None

class ChartDataResponse(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, Any]]

class SystemStats(BaseModel):
    total_requests: int
    active_users: int
    avg_response_time: float
    success_rate: float

# Mock data storage
data_storage = {}
stats = {
    "total_requests": 24800,
    "active_users": 1200,
    "avg_response_time": 42.5,
    "success_rate": 99.8
}

@app.get("/")
async def root():
    return {"message": "Welcome to Nexum Analytics API"}

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    return stats

@app.post("/api/chart-data")
async def get_chart_data(request: ChartDataRequest):
    """Get chart data for visualization"""
    # Generate mock data based on request
    labels = []
    data_points = []
    
    now = datetime.now()
    
    if request.timeframe == '1d':
        hours = 24
        labels = [(now - timedelta(hours=i)).strftime('%H:%M') for i in range(hours, 0, -1)]
        data_points = [random.uniform(500, 1500) for _ in range(hours)]
    elif request.timeframe == '7d':
        days = 7
        labels = [(now - timedelta(days=i)).strftime('%m/%d') for i in range(days, 0, -1)]
        data_points = [random.uniform(500, 2000) for _ in range(days)]
    elif request.timeframe == '30d':
        days = 30
        labels = [(now - timedelta(days=i)).strftime('%m/%d') for i in range(days, 0, -1)]
        data_points = [random.uniform(500, 2500) for _ in range(days)]
    elif request.timeframe == '90d':
        days = 90
        labels = [(now - timedelta(days=i)).strftime('%m/%d') for i in range(days, 0, -1)]
        data_points = [random.uniform(500, 3000) for _ in range(days)]
    else:
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    
    return {
        "labels": labels,
        "datasets": [{
            "label": request.metric,
            "data": data_points,
            "borderColor": "#3B82F6",
            "backgroundColor": "rgba(59, 130, 246, 0.1)",
            "fill": True
        }]
    }

@app.get("/api/data-stream")
async def stream_data():
    """Stream real-time data"""
    def data_generator():
        for i in range(10):  # Stream 10 data points
            yield {
                "timestamp": datetime.now().isoformat(),
                "value": random.uniform(100, 1000),
                "source": f"sensor_{random.randint(1, 5)}"
            }
            time.sleep(1)
    
    import time
    data = []
    for i in range(5):  # Generate 5 data points
        data.append({
            "timestamp": datetime.now().isoformat(),
            "value": random.uniform(100, 1000),
            "source": f"sensor_{random.randint(1, 5)}"
        })
        time.sleep(0.1)
    
    return {"data": data}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)