"""
FastAPI app to predict if a resume matches a job description.
"""
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Fix import path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text

# Define API models
class ResumeJobInput(BaseModel):
    resume: str
    job_description: str

class PredictionResponse(BaseModel):
    match: bool
    confidence: float
    match_score: float

# Initialize FastAPI app
app = FastAPI(
    title="Resume Screening API",
    description="API for predicting if a resume matches a job description",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and vectorizer variables globally
model = None
vectorizer = None

# Load model and vectorizer
@app.on_event("startup")
async def load_model():
    global model, vectorizer
    try:
        model = joblib.load("data/model.joblib")
        vectorizer = joblib.load("data/vectorizer.joblib")
    except FileNotFoundError:
        # For development, provide helpful error
        print("Model or vectorizer not found. Run train.py first.")
        model = None
        vectorizer = None

# For testing purposes, load the model immediately if not running as main
if __name__ != "__main__":
    try:
        model = joblib.load("data/model.joblib")
        vectorizer = joblib.load("data/vectorizer.joblib")
    except FileNotFoundError:
        print("Model or vectorizer not found. Tests may fail.")
        model = None
        vectorizer = None

# Health check endpoint
@app.get("/health")
async def health_check():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: ResumeJobInput):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Make sure to run data_generation.py and train.py first."
        )
    
    # Clean and vectorize input texts
    resume_clean = clean_text(input_data.resume)
    job_clean = clean_text(input_data.job_description)
    
    # Transform texts
    resume_vector = vectorizer.transform([resume_clean])
    job_vector = vectorizer.transform([job_clean])
    
    # Compute all features (same as in preprocessing.py)
    # 1. Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(resume_vector, job_vector)[0][0]
    
    # 2. Euclidean distance
    from sklearn.metrics.pairwise import euclidean_distances
    euclidean_dist = euclidean_distances(resume_vector.toarray(), job_vector.toarray())[0][0]
    
    # 3. Manhattan distance
    from sklearn.metrics.pairwise import manhattan_distances
    manhattan_dist = manhattan_distances(resume_vector.toarray(), job_vector.toarray())[0][0]
    
    # 4. Common terms ratio
    resume_vec = resume_vector.toarray().flatten()
    job_vec = job_vector.toarray().flatten()
    common = np.sum((resume_vec > 0) & (job_vec > 0))
    total = np.sum((resume_vec > 0) | (job_vec > 0))
    common_terms = common / total if total > 0 else 0
    
    # Combine all features
    feature = np.array([[similarity, euclidean_dist, manhattan_dist, common_terms]])
    
    # Make prediction
    prediction = model.predict(feature)[0]
    confidence = model.predict_proba(feature)[0][prediction]
    
    return {
        "match": bool(prediction),
        "confidence": float(confidence),
        "match_score": float(similarity)
    }

# Root endpoint with documentation
@app.get("/")
async def root():
    return {
        "message": "Resume Screening API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

# Run the API server
if __name__ == "__main__":
    uvicorn.run("predict_api:app", host="0.0.0.0", port=8000, reload=True)
