"""
Script to generate synthetic resumes and job descriptions using OpenAI GPT API.
"""
import os
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define roles for data generation
ROLES = [
    "Data Scientist", 
    "Machine Learning Engineer",
    "Backend Engineer", 
    "Frontend Developer",
    "DevOps Engineer",
    "Product Manager",
    "UX Designer",
    "Data Analyst",
    "Full Stack Developer",
    "AI Engineer"
]

# Define experience levels
EXPERIENCE_LEVELS = ["Entry Level", "Mid Level", "Senior Level"]

# Define prompt templates
RESUME_PROMPT = """
Generate a realistic resume for a {experience_level} {role}. Include:
1. Professional summary
2. Skills (technical and soft skills)
3. Work experience (2-3 positions)
4. Education
5. Projects (optional)

Format as plain text. Be realistic and specific.
"""

JOB_DESC_PROMPT = """
Generate a detailed job description for a {experience_level} {role} position at a tech company. Include:
1. Company overview (fictional)
2. Role responsibilities
3. Required skills and qualifications
4. Preferred skills
5. Benefits (optional)

Format as plain text. Be realistic and specific.
"""

def generate_gpt_text(prompt):
    """Generate text using OpenAI's GPT model."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating text: {e}")
        return ""

def create_dataset(num_samples=50):
    """Create a dataset of resumes and job descriptions with labels."""
    os.makedirs("data", exist_ok=True)
    
    dataset = []
    
    # Generate matching pairs (positive examples)
    for _ in range(num_samples // 2):
        role = random.choice(ROLES)
        experience = random.choice(EXPERIENCE_LEVELS)
        
        resume = generate_gpt_text(RESUME_PROMPT.format(role=role, experience_level=experience))
        job_desc = generate_gpt_text(JOB_DESC_PROMPT.format(role=role, experience_level=experience))
        
        dataset.append({
            "resume": resume,
            "job_description": job_desc,
            "role": role,
            "experience": experience,
            "match": 1  # 1 means match
        })
    
    # Generate non-matching pairs (negative examples)
    for _ in range(num_samples // 2):
        resume_role = random.choice(ROLES)
        job_role = random.choice([r for r in ROLES if r != resume_role])
        
        resume_exp = random.choice(EXPERIENCE_LEVELS)
        job_exp = random.choice(EXPERIENCE_LEVELS)
        
        resume = generate_gpt_text(RESUME_PROMPT.format(role=resume_role, experience_level=resume_exp))
        job_desc = generate_gpt_text(JOB_DESC_PROMPT.format(role=job_role, experience_level=job_exp))
        
        dataset.append({
            "resume": resume,
            "job_description": job_desc,
            "resume_role": resume_role,
            "job_role": job_role,
            "resume_experience": resume_exp,
            "job_experience": job_exp,
            "match": 0  # 0 means no match
        })
    
    # Save to file
    with open("data/resume_job_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} samples in data/resume_job_dataset.json")
    return dataset

def main():
    """Main function to generate the dataset."""
    print("Generating synthetic resume and job description data...")
    create_dataset(num_samples=30)  # Generate more samples for better training
    print("Data generation complete!")

if __name__ == "__main__":
    main()
