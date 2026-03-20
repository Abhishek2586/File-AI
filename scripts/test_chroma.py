import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
from src.modules.chromadb_handler import ChromaDBHandler
from src.modules.openai_handler import OpenAIHandler

db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chroma")
db = ChromaDBHandler(persist_directory=db_dir, collection_name="document_chunks")
openai = OpenAIHandler()

query_vector = openai.get_embedding("What is IISF?")
results = db.query(vector=query_vector, top_k=5)

for i, r in enumerate(results):
    print(f"Result {i+1}: Score = {r['score']}, File = {r['metadata'].get('source_file')}")
