from qdrant_client import QdrantClient

# Connect to the local Qdrant container
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "resume_knowledge"

def test_retrieval_suite():
    # We must set the model so QdrantClient knows how to embed our query text
    client.set_model("BAAI/bge-small-en-v1.5")
    
    test_queries = [
        "Python FastAPI and microservices backend",
        "React Native mobile development frontend",
        "Data Engineering, ETL pipelines, and PostgreSQL",
        "Machine learning models, AI, and Docker containerization",
        "Agile product management and team collaboration",
        "Cloud infrastructure deployment on AWS or GCP"
    ]
    
    import json
    all_results = {}
    
    for query_text in test_queries:
        print(f"Executing hybrid search for: '{query_text}'")
        
        results = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query_text,
            limit=3
        )
        
        query_out = []
        for i, res in enumerate(results):
            score = res.score
            meta = res.metadata
            text = meta.get("text", "Unknown Text")
            entity = meta.get("entity_name", "Unknown Entity")
            skills = meta.get("skills", [])
            query_out.append({
                "rank": i + 1,
                "score": score,
                "source": entity,
                "skills": skills,
                "text": text
            })
            
        all_results[query_text] = query_out
    
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\\nRan {len(test_queries)} test queries. Results saved to results.json")

if __name__ == "__main__":
    test_retrieval_suite()
