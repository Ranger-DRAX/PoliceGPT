"""
Pydantic Request and Response schemas for the PoliceGPT REST API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryFilter(BaseModel):
    act_name: Optional[str] = None
    act_year: Optional[int] = None
    jurisdiction: Optional[str] = "Bangladesh"


class QueryRequest(BaseModel):
    query: str = Field(..., example="দণ্ডবিধি অনুযায়ী চুরির শাস্তি কী?")
    language: str = Field("bn", description="'bn' for Bengali, 'en' for English")
    top_k: int = Field(10, description="Dense/Sparse retrieval candidates count")
    rerank_top_n: int = Field(5, description="Number of passages to rerank")
    filters: Optional[QueryFilter] = None


class RetrievedSource(BaseModel):
    chunk_id: str
    act_name_bn: Optional[str] = None
    act_name_en: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    content: str
    score: Optional[float] = None
    page_numbers: List[int] = []


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    retrieved_sources: List[RetrievedSource]
    confidence_score: float
    is_grounded: bool
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
