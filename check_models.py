import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY",""))
models = client.models.list()
print("Available models:")
for m in sorted(models.data, key=lambda x: x.id):
    print(f"  {m.id}")