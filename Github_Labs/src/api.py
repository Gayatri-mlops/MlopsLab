from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist, validator
from typing import List, Optional, Literal
from .converter import convert_gpa_to_percentage, batch_convert_gpas, compute_transcript_percentage
from .schemes import SCHEMES

app = FastAPI(title="GPA → Percentage Converter (Enhanced)", version="1.1")

SchemeName = Literal["us_4_linear", "cgpa_10_linear", "cgpa10_affine_9_5x"]

class ConvertRequest(BaseModel):
    gpa: float = Field(..., ge=0)
    scheme: SchemeName = "us_4_linear"

class ConvertResponse(BaseModel):
    percentage: float

class BatchConvertRequest(BaseModel):
    gpas: conlist(float, min_length=1)
    scheme: SchemeName = "us_4_linear"

class BatchConvertResponse(BaseModel):
    percentages: List[float]

class Course(BaseModel):
    grade: float | str
    credits: float = Field(..., gt=0)

class TranscriptRequest(BaseModel):
    courses: conlist(Course, min_length=1)
    grade_scale_max: float = Field(4.0, gt=0, description="Max of grade numbers provided (e.g., 4.0 or 10.0)")
    scheme: SchemeName = "us_4_linear"

class TranscriptResponse(BaseModel):
    weighted_gpa: float
    percentage: float

@app.get("/health")
def health():
    return {"status": "ok", "schemes": list(SCHEMES.keys())}

@app.post("/convert", response_model=ConvertResponse)
def convert(req: ConvertRequest):
    try:
        pct = convert_gpa_to_percentage(req.gpa, req.scheme)
        return {"percentage": pct}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch", response_model=BatchConvertResponse)
def batch(req: BatchConvertRequest):
    try:
        pcts = batch_convert_gpas(req.gpas, req.scheme)
        return {"percentages": pcts}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/transcript", response_model=TranscriptResponse)
def transcript(req: TranscriptRequest):
    try:
        gpa, pct = compute_transcript_percentage(
            courses=[c.model_dump() for c in req.courses],
            grade_scale_max=req.grade_scale_max,
            scheme_name=req.scheme,
        )
        return {"weighted_gpa": gpa, "percentage": pct}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))