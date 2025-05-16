"""
Preprocessing and feature extraction for resumes and job descriptions.
"""
import json
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path="data/resume_job_dataset.json"):
    """Load the dataset from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Run data_generation.py first.")
        return None

def clean_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_features(df, text_columns=['resume', 'job_description']):
    """Extract TF-IDF features from text columns."""
    # Clean text
    for col in text_columns:
        if col in df.columns:
            df[f'{col}_clean'] = df[col].apply(clean_text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,  # Ignore terms that appear in less than 2 documents
        use_idf=True,
        sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
    )
    
    # Combine all text for vocabulary building
    all_text = []
    for col in text_columns:
        clean_col = f'{col}_clean'
        if clean_col in df.columns:
            all_text.extend(df[clean_col].tolist())
    
    # Fit vectorizer on all text
    vectorizer.fit(all_text)
    
    # Transform text columns
    features = {}
    for col in text_columns:
        clean_col = f'{col}_clean'
        if clean_col in df.columns:
            features[col] = vectorizer.transform(df[clean_col])
    
    return features, vectorizer

def compute_similarity_features(resume_features, job_features):
    """Compute similarity between resume and job description vectors."""
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(resume_features, job_features)
    
    # Extract diagonal (pairwise similarities)
    similarities = np.diag(similarity_matrix)
    
    # Compute additional features
    # 1. Euclidean distance
    from sklearn.metrics.pairwise import euclidean_distances
    euclidean_dist = np.diag(euclidean_distances(resume_features.toarray(), job_features.toarray()))
    
    # 2. Manhattan distance
    from sklearn.metrics.pairwise import manhattan_distances
    manhattan_dist = np.diag(manhattan_distances(resume_features.toarray(), job_features.toarray()))
    
    # 3. Common terms ratio (non-zero elements that appear in both)
    common_terms = np.zeros(len(similarities))
    for i in range(len(similarities)):
        resume_vec = resume_features[i].toarray().flatten()
        job_vec = job_features[i].toarray().flatten()
        common = np.sum((resume_vec > 0) & (job_vec > 0))
        total = np.sum((resume_vec > 0) | (job_vec > 0))
        common_terms[i] = common / total if total > 0 else 0
    
    # Combine all features
    features = np.column_stack([
        similarities,
        euclidean_dist,
        manhattan_dist,
        common_terms
    ])
    
    return features

def prepare_dataset(df, features):
    """Prepare the final dataset for modeling."""
    # Compute similarity between resume and job description
    if 'resume' in features and 'job_description' in features:
        similarities = compute_similarity_features(
            features['resume'], 
            features['job_description']
        )
        
        # Create feature matrix
        X = similarities
        y = df['match'].values
        
        return X, y
    else:
        print("Error: Missing required features.")
        return None, None

def main():
    """Main preprocessing pipeline."""
    print("Loading and preprocessing data...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Extract features
    features, vectorizer = extract_features(df)
    
    # Prepare dataset
    X, y = prepare_dataset(df, features)
    
    if X is not None:
        print(f"Processed {len(df)} samples.")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")
    
    # Save vectorizer for later use
    import joblib
    joblib.dump(vectorizer, "data/vectorizer.joblib")
    print("Vectorizer saved to data/vectorizer.joblib")

if __name__ == "__main__":
    main()
