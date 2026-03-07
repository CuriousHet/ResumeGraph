from typing import List, Dict, Annotated
from typing_extensions import TypedDict
import operator
from pydantic import BaseModel, Field

# --- Schema Definitions for LLM Structured Output ---

class JobRequirements(BaseModel):
    """Structured requirements extracted strictly from the Job Description."""
    primary_skills: List[str] = Field(description="Must-have technical skills, languages, or tools.")
    secondary_skills: List[str] = Field(description="Nice-to-have technical skills or methodologies.")
    soft_skills: List[str] = Field(description="Required interpersonal skills like Leadership or Communication.")
    years_of_experience: str = Field(description="Expected years of experience, e.g., '3+ years', 'Entry level'.")
    key_responsibilities: List[str] = Field(description="The core tasks the candidate will perform.")
    
# --- Graph State Definition ---

class ResumeGraphState(TypedDict):
    """The central state object passed between nodes in the LangGraph workflow."""
    
    # Input
    job_description_text: str
    
    # Extraction Output
    job_requirements: JobRequirements
    
    # Retrieval Output
    retrieved_experience_bullets: List[Dict]
    retrieved_project_bullets: List[Dict]
    aligned_skills: Dict[str, List[str]]
    missing_skills: List[str]
    
    # Drafting Output
    final_resume_content: Dict
    
    # Workflow control
    errors: List[str]
