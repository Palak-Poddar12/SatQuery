from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class UploadResponse(BaseModel):
    image_ids: List[str]
    metadata: List[Dict[str, Any]]

class QueryRequest(BaseModel):
    image_ids: List[str]
    query_text: str

class ExecutionSummary(BaseModel):
    task: str
    models: List[str]
    parameters: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs_summary: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    visuals: Dict[str, Any]
    execution_summary: ExecutionSummary