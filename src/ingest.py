import os
import json
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Initialize Qdrant client connected to the local Docker container
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "resume_knowledge"

def create_collection():
    """Ensures the collection exists with the proper FastEmbed configuration for Hybrid Search."""
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        # We configure the collection for FastEmbed defaults (using BAAI/bge-small-en-v1.5)
        # FastEmbed requires specific collection settings. We can use the client's built-in fastembed methods.
        
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def extract_bullets_from_json(filepath, category, entity_name_key):
    """
    Reads a single JSON file and yields a formatted payload dictionary for every bullet point.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # We expect a list of objects (e.g. multiple experiences or projects in one file, though usually it's one)
    # The current kb might be a single object or a list. Let's handle both.
    if isinstance(data, dict):
        data = [data]
        
    for item in data:
        entity_name = item.get(entity_name_key, "Unknown")
        top_level_skills = item.get("skills", [])
        bullets = item.get("bullets", []) 
        
        for bullet_obj in bullets:
            # Depending on if 'bullet_obj' is a string or dict:
            if isinstance(bullet_obj, dict):
                text = bullet_obj.get("text", "")
                skills = bullet_obj.get("skills", top_level_skills)
            else:
                text = str(bullet_obj)
                skills = top_level_skills
                
            yield {
                "text": text,
                "category": category,
                "entity_name": entity_name,
                "skills": skills
            }

def parse_all_documents(kb_path):
    """
    Recursively finds JSON files in 'experience' and 'projects' and yields payloads.
    """
    documents = []
    
    # 1. Parse Experiences
    exp_dir = os.path.join(kb_path, "experience")
    if os.path.exists(exp_dir):
        for filename in os.listdir(exp_dir):
            if filename.endswith(".json"):
                 filepath = os.path.join(exp_dir, filename)
                 documents.extend(list(extract_bullets_from_json(filepath, category="experience", entity_name_key="company")))
                 
    # 2. Parse Projects
    proj_dir = os.path.join(kb_path, "projects")
    if os.path.exists(proj_dir):
        for filename in os.listdir(proj_dir):
            if filename.endswith(".json"):
                 filepath = os.path.join(proj_dir, filename)
                 documents.extend(list(extract_bullets_from_json(filepath, category="project", entity_name_key="name")))
                 
    return documents

def main():
    kb_path = os.path.join(os.path.dirname(__file__), "..", "resume_kb")
    
    print("Parsing Knowledge Base...")
    docs = parse_all_documents(kb_path)
    print(f"Found {len(docs)} granular bullet points.")
    
    if not docs:
        print("No documents found. Check your resume_kb directory.")
        return

    # Use Qdrant's built in FastEmbed Integration
    client.set_model("BAAI/bge-small-en-v1.5")
    
    # Recreate the collection via the set_model fastembed helper
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=client.get_fastembed_vector_params()
    )
    print(f"Re-created collection '{COLLECTION_NAME}' with dense vector config.")
    
    # FastEmbed integration in python QdrantClient allows us to use `add` 
    # instead of doing manual vectorization and `upsert`
    print("Generating embeddings and upserting directly using FastEmbed under the hood...")
    
    # We pass the metadata payloads and the strings to embed
    # Create a rich text representation for the embedding model to read
    # combining the actual text with skills and entity metadata
    embed_texts = [
        f"{doc['text']} (Skills: {', '.join(doc.get('skills', []))}) [Source: {doc.get('entity_name', '')}]"
        for doc in docs
    ]
    
    client.add(
        collection_name=COLLECTION_NAME,
        documents=embed_texts,
        metadata=[doc for doc in docs],
        ids=[uuid4().hex for _ in docs]
    )
    
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
