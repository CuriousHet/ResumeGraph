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
    
class DraftedSection(BaseModel):
    entity_name: str = Field(description="The name of the company or project, exactly as provided.")
    bullets: List[str] = Field(description="The tailored resume bullet points for this entity, rewritten to match JD keywords.")

class DraftResumeOutput(BaseModel):
    """Structured output for the drafted resume tailored to the JD."""
    experience: List[DraftedSection] = Field(description="Tailored experience sections.")
    projects: List[DraftedSection] = Field(description="Tailored project sections.")
    
class CritiqueOutput(BaseModel):
    """Structured output for fact-checking the drafted resume."""
    passed: bool = Field(description="True if the draft is factual and does not hallucinate metrics/roles. False if there are hallucinations.")
    errors: List[str] = Field(description="Specific hallucinations or additions found. Empty if passed is True.")
    ats_score: int = Field(description="A score from 0-100 indicating how well the resume matches the JD.")

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
    draft_iterations: int
    ats_score: int
