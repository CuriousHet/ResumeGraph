"""
LaTeX PDF Generator for ResumeGraph.

Takes the final output state from the LangGraph workflow, populates a Jinja2
LaTeX template, and compiles it to a PDF using pdflatex.
"""
import os
import json
import subprocess
import re
from jinja2 import Environment, FileSystemLoader


# --- LaTeX Special Character Escaping ---
LATEX_ESCAPE_MAP = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
}

def escape_latex(text: str) -> str:
    """Escapes special LaTeX characters from a string to prevent compilation errors."""
    if not isinstance(text, str):
        return str(text)
    # Use a regex to replace all special chars in one pass
    pattern = re.compile('|'.join(re.escape(key) for key in LATEX_ESCAPE_MAP.keys()))
    return pattern.sub(lambda match: LATEX_ESCAPE_MAP[match.group()], text)


# --- Load Static Data from resume_kb ---
def load_static_kb(kb_dir: str) -> dict:
    """Loads personal_info, education, skills, and publications from resume_kb."""
    data = {}
    
    # Personal Info
    pi_path = os.path.join(kb_dir, "personal_info.json")
    with open(pi_path, "r", encoding="utf-8") as f:
        pi = json.load(f)
    data["name"] = escape_latex(pi.get("name", "Your Name"))
    data["phone"] = escape_latex(pi.get("phone", ""))
    data["email"] = pi.get("email", "")  # Don't escape email, it goes into \href
    data["linkedin"] = pi.get("links", {}).get("linkedin", "")
    data["github"] = pi.get("links", {}).get("github", "")
    data["portfolio"] = pi.get("links", {}).get("portfolio", "")
    
    # Education
    edu_path = os.path.join(kb_dir, "education.json")
    with open(edu_path, "r", encoding="utf-8") as f:
        edu_data = json.load(f)
    education = []
    for entry in edu_data.get("education_history", []):
        education.append({
            "institution": escape_latex(entry.get("institution", "")),
            "degree": escape_latex(entry.get("degree", "")),
            "duration": escape_latex(entry.get("duration", "")),
            "gpa": escape_latex(entry.get("gpa", "")),
            "location": escape_latex(entry.get("location", ""))
        })
    data["education"] = education
    
    # Publications
    pub_path = os.path.join(kb_dir, "publications.json")
    with open(pub_path, "r", encoding="utf-8") as f:
        pub_data = json.load(f)
    publications = []
    for entry in pub_data.get("publications_history", []):
        publications.append({
            "title": escape_latex(entry.get("title", "")),
            "conference": escape_latex(entry.get("conference", "")),
            "abstract": escape_latex(entry.get("abstract", ""))
        })
    data["publications"] = publications
    
    return data


def load_experience_metadata(kb_dir: str) -> dict:
    """
    Loads the role and duration metadata from resume_kb/experience/*.json.
    Returns a dict keyed by company name (lowercased) for easy lookup.
    """
    metadata = {}
    exp_dir = os.path.join(kb_dir, "experience")
    if not os.path.isdir(exp_dir):
        return metadata
    for fname in os.listdir(exp_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(exp_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            entry = json.load(f)
        company = entry.get("company", "")
        metadata[company.lower()] = {
            "role": entry.get("role", "Intern"),
            "duration": entry.get("duration", ""),
            "location": entry.get("location", "")
        }
    return metadata


def load_project_metadata(kb_dir: str) -> dict:
    """
    Loads skills metadata from resume_kb/projects/*.json.
    Returns a dict keyed by project name (lowercased) for easy lookup.
    """
    metadata = {}
    proj_dir = os.path.join(kb_dir, "projects")
    if not os.path.isdir(proj_dir):
        return metadata
    for fname in os.listdir(proj_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(proj_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            entry = json.load(f)
        name = entry.get("name", entry.get("project_name", ""))
        metadata[name.lower()] = {
            "skills": entry.get("skills", [])
        }
    return metadata


def build_template_context(final_state: dict, kb_dir: str) -> dict:
    """
    Merges the LangGraph final state (draft_resume output) with static KB data
    to produce a complete template context dictionary.
    """
    # Load static sections
    context = load_static_kb(kb_dir)
    
    # Load metadata lookups
    exp_meta = load_experience_metadata(kb_dir)
    proj_meta = load_project_metadata(kb_dir)
    
    # Process drafted experience
    draft = final_state.get("final_resume_content", {})
    experience_sections = []
    for exp in draft.get("experience", []):
        entity = exp.get("entity_name", "Unknown")
        meta = exp_meta.get(entity.lower(), {})
        experience_sections.append({
            "entity_name": escape_latex(entity),
            "role": escape_latex(meta.get("role", "Intern")),
            "duration": escape_latex(meta.get("duration", "")),
            "location": escape_latex(meta.get("location", "")),
            "bullets": [escape_latex(b) for b in exp.get("bullets", [])]
        })
    context["experience"] = experience_sections
    
    # Process drafted projects
    project_sections = []
    for proj in draft.get("projects", []):
        entity = proj.get("entity_name", "Unknown")
        meta = proj_meta.get(entity.lower(), {})
        project_sections.append({
            "entity_name": escape_latex(entity),
            "skills": [escape_latex(s) for s in meta.get("skills", [])],
            "bullets": [escape_latex(b) for b in proj.get("bullets", [])]
        })
    context["projects"] = project_sections
    
    # Process skills (use aligned_skills from the retrieval node if available)
    aligned = final_state.get("aligned_skills", {})
    if aligned:
        # Format category names nicely
        formatted_skills = {}
        category_name_map = {
            "languages": "Languages",
            "ai_ml": "AI \\& Machine Learning",
            "cloud": "Cloud \\& DevOps",
            "databases": "Web \\& Databases",
        }
        for cat, skills_list in aligned.items():
            display_name = category_name_map.get(cat, escape_latex(cat.replace("_", " ").title()))
            formatted_skills[display_name] = [escape_latex(s) for s in skills_list]
        context["skills"] = formatted_skills
    else:
        # Fallback: load raw skills from KB
        skills_path = os.path.join(kb_dir, "skills.json")
        with open(skills_path, "r", encoding="utf-8") as f:
            raw_skills = json.load(f)
        context["skills"] = {k.replace("_", " ").title(): v for k, v in raw_skills.items()}
    
    return context


def render_latex(context: dict, template_dir: str) -> str:
    """
    Renders the Jinja2 LaTeX template with the given context.
    Uses custom delimiters to avoid clashing with LaTeX curly braces.
    """
    env = Environment(
        loader=FileSystemLoader(template_dir),
        block_start_string='<%',
        block_end_string='%>',
        variable_start_string='<<',
        variable_end_string='>>',
        comment_start_string='<#',
        comment_end_string='#>',
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("jinja_template.tex")
    return template.render(**context)


def compile_pdf(tex_content: str, output_dir: str, filename: str = "final_resume") -> str:
    """
    Writes the rendered LaTeX to a .tex file and compiles it into a PDF
    using pdflatex. Returns the path to the generated PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tex_path = os.path.join(output_dir, f"{filename}.tex")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    
    print(f"Wrote LaTeX source to: {tex_path}")
    
    # Run pdflatex twice (needed for references/page layout to stabilize)
    for i in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", output_dir, tex_path],
            capture_output=True,
            text=True,
            cwd=output_dir
        )
        if result.returncode != 0:
            print(f"pdflatex pass {i+1} returned code {result.returncode}")
            # Print last 30 lines of the log for debugging
            log_lines = result.stdout.split('\n')
            print("--- pdflatex output (last 30 lines) ---")
            for line in log_lines[-30:]:
                print(line)
            if i == 0:
                print("Attempting second pass anyway...")
            else:
                print("WARNING: PDF compilation may have issues. Check the .log file.")
    
    if os.path.exists(pdf_path):
        print(f"\n✅ PDF generated successfully: {pdf_path}")
    else:
        print(f"\n❌ PDF was not generated. Check {output_dir}/{filename}.log for errors.")
    
    return pdf_path


def generate_resume_pdf(final_state: dict, project_root: str = None) -> str:
    """
    Main entry point. Takes the final LangGraph state and produces a PDF resume.
    
    Args:
        final_state: The final state dict from graph.invoke()
        project_root: The root directory of the ResumeGraph project
        
    Returns:
        Path to the generated PDF file
    """
    if project_root is None:
        project_root = os.path.join(os.path.dirname(__file__), "..")
    
    project_root = os.path.abspath(project_root)
    kb_dir = os.path.join(project_root, "resume_kb")
    template_dir = os.path.join(project_root, "latex_template")
    output_dir = os.path.join(project_root, "out")
    
    print("\n--- GENERATING PDF ---")
    
    # 1. Build the full template context
    context = build_template_context(final_state, kb_dir)
    print(f"Context built: {len(context['experience'])} experiences, {len(context['projects'])} projects")
    
    # 2. Render the Jinja2 template
    tex_content = render_latex(context, template_dir)
    print(f"LaTeX rendered: {len(tex_content)} characters")
    
    # 3. Compile to PDF
    pdf_path = compile_pdf(tex_content, output_dir)
    
    return pdf_path
