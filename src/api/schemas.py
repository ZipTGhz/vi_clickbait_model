from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    title: str
    lead_paragraph: Optional[str] = ''

class PredictResponse(BaseModel):
    label: str
    confidence: float
    is_clickbait: bool
