import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    max_retries=1,
    timeout=5.0
)
try:
    print("Testing Embedding API...")
    resp = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=["What is IISF?"]
    )
    print("Success. Embedding length: ", len(resp.data[0].embedding))
except Exception as e:
    print(f"Error: {e}")
