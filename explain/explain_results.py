import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def explain_medical_report(text: str, model: str = "llama3") -> str:
    prompt = f"""
You are a medical expert AI. Based on the lab report below, explain it in simple terms for a patient. Include any abnormalities and what they mean.

Lab Report:
{text}
"""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })

    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        return f"❌ Error from Ollama: {response.text}"
