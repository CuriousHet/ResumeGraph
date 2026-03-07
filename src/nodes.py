import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import ResumeGraphState, JobRequirements

def extract_jd_requirements(state: ResumeGraphState) -> ResumeGraphState:
    """
    Node 1: Parses the raw Job Description and extracts a structured set of requirements
    using Gemini 2.5 Flash.
    """
    print("--- NODE 1: EXTRACTING JD REQUIREMENTS ---")
    
    jd_text = state.get("job_description_text", "")
    if not jd_text:
        return {"errors": ["No job description provided."]}
        
    # Ensure the API key exists
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
         print("WARNING: GOOGLE_API_KEY not found in environment. The LLM call will likely fail.")
    
    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0  # Zero temperature for deterministic extraction
    )
    
    # Bind the LLM to output our exact Pydantic schema
    structured_llm = llm.with_structured_output(JobRequirements)
    
    prompt = f"""
    You are an expert technical recruiter analyzing a job description.
    Extract the core requirements from the following job description and categorize them exactly as requested.
    
    JOB DESCRIPTION:
    {jd_text}
    """
    
    try:
        # Call the model
        extracted_reqs = structured_llm.invoke(prompt)
        print(f"Extraction Successful. Found {len(extracted_reqs.primary_skills)} primary skills.")
        
        # Return the partial state update
        return {"job_requirements": extracted_reqs}
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return {"errors": [f"JD Extraction failed: {str(e)}"]}
