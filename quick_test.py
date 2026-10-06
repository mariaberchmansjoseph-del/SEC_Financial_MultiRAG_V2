import os
import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
print(f"Key starts with: {GROQ_KEY[:8]}...")

# Try with custom HTTP client (bypasses SSL issues)
try:
    http_client = httpx.Client(verify=False)
    client = Groq(
        api_key     = GROQ_KEY,
        http_client = http_client
    )
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [
            {"role": "user",
             "content": "Say hello in one word."}
        ],
        max_tokens = 10
    )
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Try with no SSL verification
print()
print("Trying requests library...")
try:
    import requests
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        model = "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": "Say hello."}
        ],
        "max_tokens": 10
    }
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers = headers,
        json    = data,
        verify  = False,
        timeout = 30
    )
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")

except Exception as e:
    print(f"requests ERROR: {e}")