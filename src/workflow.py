from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

from src.state import ResumeGraphState
from src.nodes import extract_jd_requirements, retrieve_matching_bullets, draft_resume, critique_and_fact_check

# Load environment variables (like GOOGLE_API_KEY) from .env
load_dotenv()

def route_after_critique(state: ResumeGraphState) -> str:
    """Always stop after one iteration as requested."""
    return "end"

def build_graph():
    """Builds the LangGraph workflow."""
    
    # Initialize the Graph with our typed state
    workflow = StateGraph(ResumeGraphState)
    
    # Add nodes
    workflow.add_node("extract_jd", extract_jd_requirements)
    workflow.add_node("retrieve_bullets", retrieve_matching_bullets)
    workflow.add_node("draft_resume", draft_resume)
    workflow.add_node("critique_and_fact_check", critique_and_fact_check)
    
    # Set the entry point
    workflow.set_entry_point("extract_jd")
    
    # Chain them together
    workflow.add_edge("extract_jd", "retrieve_bullets")
    workflow.add_edge("retrieve_bullets", "draft_resume")
    workflow.add_edge("draft_resume", "critique_and_fact_check")
    
    workflow.add_conditional_edges(
        "critique_and_fact_check",
        route_after_critique,
        {
            "draft_resume": "draft_resume",
            "end": END
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    return app

def run_single_jd(graph, jd_text: str, jd_name: str, project_root: str):
    """Runs the full pipeline for a single JD and generates a named PDF."""
    print(f"\n{'='*60}")
    print(f"  PROCESSING: {jd_name}")
    print(f"{'='*60}")

    initial_state = {"job_description_text": jd_text}
    final_state = graph.invoke(initial_state)

    # --- Print summary ---
    reqs = final_state.get("job_requirements")
    if reqs:
        print(f"  Extracted {len(reqs.primary_skills)} primary skills.")

    draft = final_state.get("final_resume_content")
    errors = final_state.get("errors", [])
    ats_score = final_state.get("ats_score", 0)

    # --- Save Critiques ---
    critique_dir = os.path.join(project_root, "out", "critiques")
    os.makedirs(critique_dir, exist_ok=True)
    critique_path = os.path.join(critique_dir, f"{jd_name}.json")
    
    import json
    with open(critique_path, "w", encoding="utf-8") as f:
        json.dump({
            "ats_score": ats_score,
            "hallucinations": errors,
            "passed": len(errors) == 0
        }, f, indent=2)
    print(f"  Critique saved to: {critique_path}")

    if draft:
        from src.generate_pdf import generate_resume_pdf
        pdf_path = generate_resume_pdf(final_state, project_root, filename=jd_name)
        print(f"  ✅ Resume saved to: {pdf_path}")
        return pdf_path
    else:
        print(f"  ❌ Skipped PDF generation (no draft generated).")
        return None


if __name__ == "__main__":
    import glob

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    jd_input_dir = os.path.join(project_root, "jd_inputs")

    # Collect JD files: prefer jd_inputs/ directory, fall back to sample_jd.txt
    jd_files = []
    if os.path.isdir(jd_input_dir):
        jd_files = sorted(glob.glob(os.path.join(jd_input_dir, "*.txt")))

    if not jd_files:
        # Fallback to the single sample JD
        fallback = os.path.join(project_root, "sample_jd.txt")
        if os.path.exists(fallback):
            jd_files = [fallback]
        else:
            print("No JD files found. Place .txt files in jd_inputs/ or provide sample_jd.txt")
            exit(1)

    print(f"\nFound {len(jd_files)} JD(s) to process.")

    graph = build_graph()
    results = []

    for jd_path in jd_files:
        jd_name = os.path.splitext(os.path.basename(jd_path))[0]
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
        pdf = run_single_jd(graph, jd_text, jd_name, project_root)
        results.append((jd_name, pdf))

    # --- Final Summary ---
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {len(results)} JD(s) processed")
    print(f"{'='*60}")
    for name, pdf in results:
        status = f"✅ {pdf}" if pdf else "❌ Failed"
        print(f"  {name}: {status}")
