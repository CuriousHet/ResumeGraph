from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

from src.state import ResumeGraphState
from src.nodes import extract_jd_requirements

# Load environment variables (like GOOGLE_API_KEY) from .env
load_dotenv()

def build_graph():
    """Builds the LangGraph workflow."""
    
    # Initialize the Graph with our typed state
    workflow = StateGraph(ResumeGraphState)
    
    # Add Node 1
    workflow.add_node("extract_jd", extract_jd_requirements)
    
    # Set the entry point
    workflow.set_entry_point("extract_jd")
    
    # For now, end the graph after extraction so we can test it step-by-step
    workflow.add_edge("extract_jd", END)
    
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
        print(f"Primary Skills: {reqs.primary_skills}")
        print(f"Secondary Skills: {reqs.secondary_skills}")
        print(f"Soft Skills: {reqs.soft_skills}")
        print(f"Years of Exp: {reqs.years_of_experience}")
    
    errors = final_state.get("errors")
    if errors:
        print(f"ERRORS ENCOUNTERED: {errors}")
