"""
Tests for the preprocessing module.
"""
import pytest
import numpy as np
from src.preprocessing import clean_text, extract_features, compute_similarity_features
import pandas as pd

def test_clean_text():
    """Test text cleaning function."""
    text = "This is a TEST with Punctuation!!! And some Numbers: 123."
    cleaned = clean_text(text)
    assert cleaned == "this is a test with punctuation and some numbers 123"
    
    # Test with non-string input
    assert clean_text(None) == ""
    assert clean_text(123) == ""

def test_extract_features():
    """Test feature extraction from text."""
    # Create a simple test dataframe
    df = pd.DataFrame({
        'resume': [
            "Python developer with 5 years experience in machine learning",
            "Frontend developer with React and JavaScript experience"
        ],
        'job_description': [
            "Looking for Python developer with machine learning skills",
            "Senior JavaScript developer needed for web application"
        ]
    })
    
    # Extract features
    features, vectorizer = extract_features(df)
    
    # Check that features were extracted for both columns
    assert 'resume' in features
    assert 'job_description' in features
    
    # Check dimensions
    assert features['resume'].shape[0] == 2  # 2 samples
    assert features['job_description'].shape[0] == 2  # 2 samples
    
    # Check that vectorizer was created
    assert vectorizer is not None

def test_compute_similarity_features():
    """Test similarity feature computation."""
    # Create simple test vectors
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer()
    resume_vectors = vectorizer.fit_transform([
        "Python developer with machine learning experience",
        "Frontend developer with JavaScript experience"
    ])
    job_vectors = vectorizer.transform([
        "Looking for Python developer with ML skills",
        "JavaScript developer needed for web application"
    ])
    
    # Compute similarity features
    features = compute_similarity_features(resume_vectors, job_vectors)
    
    # Check dimensions
    assert features.shape == (2, 4)  # 2 samples, 4 features
    
    # Check feature values are within expected ranges
    # Cosine similarity: between -1 and 1
    assert np.all(features[:, 0] >= -1) and np.all(features[:, 0] <= 1)
    
    # Euclidean and Manhattan distances: non-negative
    assert np.all(features[:, 1] >= 0)
    assert np.all(features[:, 2] >= 0)
    
    # Common terms ratio: between 0 and 1
    assert np.all(features[:, 3] >= 0) and np.all(features[:, 3] <= 1)
