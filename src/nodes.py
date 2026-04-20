import os
from src.state import ResumeGraphState, JobRequirements, DraftResumeOutput, CritiqueOutput
from src.key_manager import get_key_manager
from langchain_core.prompts import PromptTemplate

# --- Model Configuration ---
# Assign optimal model per node based on task complexity.
MODEL_EXTRACT = "gemini-2.5-flash"      
MODEL_DRAFT   = "gemini-2.5-flash"         
MODEL_CRITIQUE = "gemini-2.5-flash"  

def draft_resume(state: ResumeGraphState) -> ResumeGraphState:
    """
    Node 3: Takes the filtered, grouped bullets from Node 2 and rewrites them
    to perfectly align with the JobRequirements (Node 1) without hallucinating.
    """
    print("--- NODE 3: DRAFTING RESUME ---")
    
    reqs = state.get("job_requirements")
    exps = state.get("retrieved_experience_bullets", [])
    projs = state.get("retrieved_project_bullets", [])
    aligned = state.get("aligned_skills", {})
    errors = state.get("errors", [])
    
    if not exps and not projs:
        return {"errors": ["No bullets were retrieved to draft."]}
        
    km = get_key_manager()
    
    error_prompt = ""
    if errors:
        error_prompt = f"\nCRITICAL: PREVIOUS DRAFT HAD HALLUCINATIONS. FIX THESE: {errors}\n"
    
    prompt = f"""
    You are a senior technical resume writer specializing in software engineering roles.
    Your output will be pasted DIRECTLY into a resume — zero manual editing should be required.

    ════════════════════════════════════════
    HARD CONSTRAINTS  (violations = reject output)
    ════════════════════════════════════════
    - Exactly 3 bullets per experience entry.
    - Exactly 3 bullets per project entry.
    - Exactly the top 4 most relevant projects (ranked by JD alignment).
    - Every bullet: 30–40 words. Count before finalising.
    - Every bullet starts with a PAST-TENSE ACTION VERB (Built, Designed, Reduced, Led, Migrated…).
    - Mention 1–3 technologies per bullet — never front-load a laundry list.
    - NEVER invent metrics, tools, or responsibilities not present in the source data.
    - NEVER use filler phrases: "contributed to", "worked on", "helped with", "assisted in".
    - Process ALL experience entries — do not skip any.
    {error_prompt}

    ════════════════════════════════════════
    BULLET STRUCTURE (follow in order for every entity)
    ════════════════════════════════════════
    Bullet 1 → WHAT was built / problem solved + primary technology.
    Bullet 2 → HOW: specific engineering approach, design decision, or key challenge overcome.
    Bullet 3 → OUTCOME: business value, performance gain, or user impact (use metric only if in source data).

    ════════════════════════════════════════
    JD KEYWORD INJECTION RULES
    ════════════════════════════════════════
    - Use JD keywords only when they naturally describe what was actually done.
    - Prefer JD vocabulary over synonyms when both are equally accurate.
    - Primary skills take priority over secondary skills.
    - Do NOT force-fit a keyword that distorts the original meaning.

    Primary Skills      : {reqs.primary_skills}
    Secondary Skills    : {reqs.secondary_skills}
    Key Responsibilities: {reqs.key_responsibilities}

    ════════════════════════════════════════
    SOURCE DATA
    ════════════════════════════════════════
    Legend: 'fact' = raw technical detail | 'impact' = proven result | 'bullet' = pre-written draft

    EXPERIENCES (process every entry):
    {exps}

    PROJECTS (select top 4 by JD relevance, then process each):
    {projs}

    ════════════════════════════════════════
    OUTPUT FORMAT  — follow exactly, no extra commentary
    ════════════════════════════════════════
    Return ONLY the two sections below. No preamble, no explanation, no markdown headers beyond what is shown.

    EXPERIENCE
    <Company / Role / Dates as given in source>
    - <bullet 1>
    - <bullet 2>
    - <bullet 3>

    [repeat for every experience entry]

    PROJECTS
    <Project Name as given in source>
    - <bullet 1>
    - <bullet 2>
    - <bullet 3>

    [repeat for top 4 projects]

    ════════════════════════════════════════
    SELF-CHECK before returning output
    ════════════════════════════════════════
    Silently verify:
    ✓ Word count 30–40 for every bullet.
    ✓ No invented data.
    ✓ All experiences included.
    ✓ Exactly 4 projects.
    ✓ No filler phrases.
    ✓ Every bullet opens with a past-tense verb.
    Only output the final result after passing all checks.
    """
    
    try:
        draft = km.invoke_with_retry(
            model=MODEL_DRAFT,
            temperature=0.3,
            prompt=prompt,
            structured_output_schema=DraftResumeOutput,
        )
        print(f"Drafting Successful. Drafted {len(draft.experience)} experiences and {len(draft.projects)} projects.")
        
        # Convert Pydantic object back to a standard dictionary to store in state
        draft_dict = draft.dict()
        return {"final_resume_content": draft_dict, "draft_iterations": state.get("draft_iterations", 0) + 1}
        
    except Exception as e:
        print(f"Error during drafting: {e}")
        return {"errors": [f"Drafting failed: {str(e)}"]}

def critique_and_fact_check(state: ResumeGraphState) -> ResumeGraphState:
    """
    Node 4: Compares the DraftResumeOutput against the original retrieved bullets.
    Flags any added metrics or responsibilities not present in the original data.
    """
    print("--- NODE 4: CRITIQUE AND FACT-CHECK ---")
    
    draft = state.get("final_resume_content", {})
    exps = state.get("retrieved_experience_bullets", [])
    projs = state.get("retrieved_project_bullets", [])
    
    if not draft:
        return {"errors": ["No drafted resume content found to critique."]}
        
    km = get_key_manager()
    
    prompt = f"""
    You are an expert strict Fact-Checker. 
    Compare the newly drafted resume sections against the original retrieved data.
    
    ORIGINAL EXPERIENCE DATA: {exps}
    ORIGINAL PROJECT DATA: {projs}
    
    Note: These are categorized as 'fact' (raw technical details), 'impact' (business results), or 'bullet' (pre-written points).
    
    DRAFTED RESUME: {draft}
    
    RULES:
    1. If the draft contains ANY numbers, metrics, or technologies that are NOT explicitly mentioned in the ORIGINAL data (checking across facts, impacts, and bullets), it is a hallucination.
    2. If the draft invents a responsibility not implied by the original data, it is a hallucination.
    3. ATS SCORE: Evaluate how well the drafted resume matches the Job Description. Assign a score from 0 to 100.
    4. If there are no hallucinations, set 'passed' to True and 'errors' to an empty list.
    5. If there are hallucinations, set 'passed' to False and list EXACTLY what was hallucinated in 'errors'.
    """
    
    try:
        critique = km.invoke_with_retry(
            model=MODEL_CRITIQUE,
            temperature=0.0,
            prompt=prompt,
            structured_output_schema=CritiqueOutput,
        )
        
        if critique.passed:
            print(f"Fact-Check passed! ATS Score: {critique.ats_score}. No hallucinations detected.")
            return {"errors": [], "ats_score": critique.ats_score}
        else:
            print(f"Fact-Check failed. ATS Score: {critique.ats_score}. Found {len(critique.errors)} hallucinations.")
            return {"errors": critique.errors, "ats_score": critique.ats_score}
            
    except Exception as e:
        print(f"Error during critique: {e}")
        return {"errors": [f"Critique failed: {str(e)}"]}

def extract_jd_requirements(state: ResumeGraphState) -> ResumeGraphState:
    """
    Node 1: Parses the raw Job Description and extracts a structured set of requirements
    using Gemini 2.5 Flash.
    """
    print("--- NODE 1: EXTRACTING JD REQUIREMENTS ---")
    
    jd_text = state.get("job_description_text", "")
    if not jd_text:
        return {"errors": ["No job description provided."]}
        
    km = get_key_manager()
    
    prompt = f"""
    You are an expert technical recruiter analyzing a job description.
    Extract the core requirements from the following job description and categorize them exactly as requested.
    
    JOB DESCRIPTION:
    {jd_text}
    """
    
    try:
        # Call the model with key rotation
        extracted_reqs = km.invoke_with_retry(
            model=MODEL_EXTRACT,
            temperature=0.0,
            prompt=prompt,
            structured_output_schema=JobRequirements,
        )
        print(f"Extraction Successful. Found {len(extracted_reqs.primary_skills)} primary skills.")
        
        # Return the partial state update
        return {"job_requirements": extracted_reqs}
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return {"errors": [f"JD Extraction failed: {str(e)}"]}

def retrieve_matching_bullets(state: ResumeGraphState) -> ResumeGraphState:
    """
    Node 2: Takes the extracted job requirements and queries Qdrant for the best
    matching resume bullet points from the KB.
    """
    print("--- NODE 2: HYBRID RETRIEVAL ---")
    
    reqs = state.get("job_requirements")
    if not reqs:
        return {"errors": ["No job requirements found to query against."]}
        
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        client.set_model("BAAI/bge-small-en-v1.5")
        
        # Build a search query combining all technical skills
        search_query = " ".join(reqs.primary_skills + reqs.secondary_skills)
        print(f"Querying Qdrant for: {search_query}")
        
        # We want to retrieve a large pool to then filter down (e.g. limit 20)
        exp_results = client.query(
            collection_name="resume_knowledge",
            query_text=search_query,
            query_filter={"must": [{"key": "category", "match": {"value": "experience"}}]},
            limit=20
        )
        
        # Query for Projects
        proj_results = client.query(
            collection_name="resume_knowledge",
            query_text=search_query,
            query_filter={"must": [{"key": "category", "match": {"value": "project"}}]},
            limit=15
        )
        
        # Group Experience bullets by entity (Company), taking top 5
        grouped_exps = {}
        for res in exp_results:
            entity = res.metadata.get("entity_name", "Unknown")
            if entity not in grouped_exps:
                grouped_exps[entity] = []
            if len(grouped_exps[entity]) < 5: # max 5 per company
                grouped_exps[entity].append({
                    "text": res.metadata.get("text", ""),
                    "type": res.metadata.get("sub_category", "bullet"),
                    "score": res.score
                })
                
        # Group Project bullets by entity (Project Name), taking top 5
        grouped_projs = {}
        for res in proj_results:
            entity = res.metadata.get("entity_name", "Unknown")
            if entity not in grouped_projs:
                grouped_projs[entity] = []
            if len(grouped_projs[entity]) < 5: # max 5 per project
                grouped_projs[entity].append({
                    "text": res.metadata.get("text", ""),
                    "type": res.metadata.get("sub_category", "bullet"),
                    "score": res.score
                })
        
        # Format the output back to a list of dicts for the LLM to rewrite easily
        retrieved_exps = [{"entity_name": k, "data": v} for k, v in grouped_exps.items()]
        retrieved_projs = [{"entity_name": k, "data": v} for k, v in grouped_projs.items()]
        
        print(f"Grouped to {len(retrieved_exps)} companies and {len(retrieved_projs)} projects.")
        
        # Process Skills File
        import json
        import os
        
        skills_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resume_kb", "skills.json")
        aligned_skills = {}
        missing_skills = []
        
        if os.path.exists(skills_path):
            with open(skills_path, "r", encoding="utf-8") as f:
                core_skills = json.load(f)
                
            jd_skills_set = set(reqs.primary_skills + reqs.secondary_skills)
            jd_skills_lower = {s.lower() for s in jd_skills_set}
            
            # Reorder categories putting JD priority skills first
            for category, skills_list in core_skills.items():
                matched = [s for s in skills_list if s.lower() in jd_skills_lower]
                unmatched = [s for s in skills_list if s.lower() not in jd_skills_lower]
                aligned_skills[category] = matched + unmatched
                
            # Find missing skills that were in the JD but not our KB skills.json
            all_kb_skills_lower = {s.lower() for cats in core_skills.values() for s in cats}
            missing_skills = [s for s in jd_skills_set if s.lower() not in all_kb_skills_lower]
            
        else:
            print("WARNING: skills.json not found.")

        return {
            "retrieved_experience_bullets": retrieved_exps,
            "retrieved_project_bullets": retrieved_projs,
            "aligned_skills": aligned_skills,
            "missing_skills": missing_skills
        }
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {"errors": [f"Retrieval failed: {str(e)}"]}

