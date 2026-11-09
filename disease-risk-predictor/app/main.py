import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from risk.features import FEATURES

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
DISCLAIMER = ("Educational demo trained on a small public dataset. "
              "It is not a medical device and does not replace a doctor.")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", ROOT / "model" / "model.joblib"))
    if not path.exists():
        raise RuntimeError(f"Model not found at {path}. Run: python -m risk.train")
    state["bundle"] = joblib.load(path)
    yield
    state.clear()


app = FastAPI(
    title="Diabetes Risk Predictor",
    description="Estimates diabetes risk from basic health measurements (Pima Indians dataset).",
    version="1.0.0",
    lifespan=lifespan,
)


class Patient(BaseModel):
    """Leave a measurement out (null) if it is unknown: it is replaced by a typical value."""
    Pregnancies: int = Field(ge=0, le=20, description="Number of pregnancies")
    Glucose: Optional[float] = Field(None, ge=20, le=300, description="Plasma glucose (mg/dL)")
    BloodPressure: Optional[float] = Field(None, ge=20, le=200, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: Optional[float] = Field(None, ge=1, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: Optional[float] = Field(None, ge=1, le=900, description="2-hour serum insulin (mu U/ml)")
    BMI: Optional[float] = Field(None, ge=10, le=70, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(ge=0.0, le=2.5, description="Family history score")
    Age: int = Field(ge=18, le=100, description="Age in years")

    model_config = {"json_schema_extra": {"example": {
        "Pregnancies": 2, "Glucose": 138, "BloodPressure": 72, "SkinThickness": 30,
        "Insulin": 120, "BMI": 33.6, "DiabetesPedigreeFunction": 0.63, "Age": 45}}}


class Prediction(BaseModel):
    probability: float
    risk_level: str
    missing_inputs: list[str]
    model: str
    disclaimer: str


def risk_level(p: float) -> str:
    return "low" if p < 0.33 else "moderate" if p < 0.66 else "high"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "bundle" in state}


@app.get("/model-info")
def model_info():
    info = state["bundle"]["info"]
    return {k: info[k] for k in
            ("model_name", "trained_at", "dataset", "test_metrics", "comparison", "feature_importance")}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    bundle = state["bundle"]
    values = patient.model_dump()
    missing = [f for f in FEATURES if values[f] is None]
    row = pd.DataFrame([[np.nan if values[f] is None else values[f] for f in FEATURES]], columns=FEATURES)
    try:
        p = float(bundle["pipeline"].predict_proba(row)[0, 1])
    except Exception as e:  # should not happen with validated input
        raise HTTPException(status_code=500, detail="Prediction failed.") from e
    return Prediction(
        probability=round(p, 4),
        risk_level=risk_level(p),
        missing_inputs=missing,
        model=bundle["info"]["model_name"],
        disclaimer=DISCLAIMER,
    )


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
