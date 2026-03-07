from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

from src.state import ResumeGraphState
from src.nodes import extract_jd_requirements, retrieve_matching_bullets

# Load environment variables (like GOOGLE_API_KEY) from .env
load_dotenv()

def build_graph():
    """Builds the LangGraph workflow."""
    
    # Initialize the Graph with our typed state
    workflow = StateGraph(ResumeGraphState)
    
    # Add Node 1 and Node 2
    workflow.add_node("extract_jd", extract_jd_requirements)
    workflow.add_node("retrieve_bullets", retrieve_matching_bullets)
    
    # Set the entry point
    workflow.set_entry_point("extract_jd")
    
    # Chain them together
    workflow.add_edge("extract_jd", "retrieve_bullets")
    workflow.add_edge("retrieve_bullets", END)
    
    # Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    
    print("\n--- Testing LangGraph Node 1: Extraction ---")
    
    # Load the sample JD
    jd_path = os.path.join(os.path.dirname(__file__), "..", "sample_jd.txt")
    with open(jd_path, "r", encoding="utf-8") as f:
        sample_jd = f.read()
        
    # Initialize state
    initial_state = {
        "job_description_text": sample_jd
    }
    
    graph = build_graph()
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    # Print the output nicely
    print("\n--- FINAL OUTPUT STATE ---")
    
    reqs = final_state.get("job_requirements")
    if reqs:
        print(f"\n[Extraction Node Outputs]")
        print(f"Primary Skills: {reqs.primary_skills}")
        print(f"Years of Exp: {reqs.years_of_experience}")
        
    exps = final_state.get("retrieved_experience_bullets")
    projs = final_state.get("retrieved_project_bullets")
    
    if exps or projs:
        print(f"\n[Retrieval Node Outputs]")
        print(f"Experience Companies: {len(exps)}")
        if len(exps) > 0:
            print(f"  #1: {exps[0]['entity_name']} with {len(exps[0]['bullets'])} bullets")
            print(f"    - {exps[0]['bullets'][0]['text']}")
            
        print(f"\nProject Entities: {len(projs)}")
        if len(projs) > 0:
            print(f"  #1: {projs[0]['entity_name']} with {len(projs[0]['bullets'])} bullets")
            print(f"    - {projs[0]['bullets'][0]['text']}")
            
    aligned = final_state.get("aligned_skills", {})
    missing = final_state.get("missing_skills", [])
    if aligned:
        print(f"\n[Skills Output]")
        for category, skills in aligned.items():
            print(f"  {category.capitalize()}: {', '.join(skills)}")
        print(f"  Missing Found in JD: {', '.join(missing)}")
    
    errors = final_state.get("errors")
    if errors:
        print(f"\nERRORS ENCOUNTERED: {errors}")
