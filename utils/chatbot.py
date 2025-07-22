# utils/chatbot.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def get_chat_response(
    question: str,
    context: str,
    model: str = "llama3",
    verbose: bool = False
) -> str:
    """
    Generate a chat response using the selected LLM model based on user question and medical context.

    Parameters:
        question (str): The user’s follow-up question.
        context (str): Cleaned medical report text.
        model (str): Model to use (default is "llama3").
        verbose (bool): If True, print debug output.

    Returns:
        str: AI's response or error message.
    """

    prompt = f"""
You are a medical assistant AI. The following is a medical report uploaded by a patient:

\"\"\"{context}\"\"\"

Based on this report, answer the patient's question below in a simple and helpful way:

Q: {question}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=20  # seconds
        )

        if verbose:
            print(f"Prompt:\n{prompt}")
            print(f"Response Code: {response.status_code}")
            print(f"Response: {response.text}")

        if response.status_code == 200:
            reply = response.json().get("response", "").strip()
            return reply if reply else "🤖 I couldn't generate a response. Please try again."
        else:
            return f"❌ Error from Ollama ({response.status_code}): {response.text}"

    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error: {str(e)}"
