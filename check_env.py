import os
from dotenv import load_dotenv
from pathlib import Path

# Show where .env is being loaded from
env_path = Path(".env")
print(f".env exists: {env_path.exists()}")
print(f".env path: {env_path.absolute()}")

load_dotenv()

key = os.getenv("GROQ_API_KEY", "NOT FOUND")
print(f"Key value: '{key}'")
print(f"Key length: {len(key)}")