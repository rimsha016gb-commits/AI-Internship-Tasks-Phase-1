# ============================================================
# Task 4: General Health Query Chatbot (Prompt Engineering)
# DevelopersHub Corporation – AI/ML Engineering Internship
# Tool: Groq API (Free)
# ============================================================

from groq import Groq

# ── 1. Setup Groq Client ─────────────────────────────────────
# Replace with your actual Groq API key
# Get free key from: https://console.groq.com/keys
client = Groq(api_key="Your_API_Key")

# ── 2. System Prompt (Prompt Engineering) ────────────────────
SYSTEM_PROMPT = """
You are a friendly and knowledgeable health assistant.
Your job is to answer general health-related questions in a
clear, simple, and easy-to-understand way.

Follow these rules strictly:
1. Give helpful and accurate general health information.
2. Always be warm, friendly, and supportive in tone.
3. NEVER diagnose a disease or prescribe medication.
4. NEVER give specific dosage instructions for any medicine.
5. For any serious symptoms, always recommend seeing a doctor.
6. Keep your answers concise — 3 to 5 sentences maximum.
7. End every response with: "Please consult a doctor for
   personalized medical advice."
"""

# ── 3. Safety Filter ─────────────────────────────────────────
UNSAFE_KEYWORDS = [
    "suicide", "kill myself", "overdose", "how to die",
    "self harm", "cut myself", "end my life", "poison",
    "drug abuse", "illegal drugs"
]

def is_unsafe(user_input):
    """Check if user query contains unsafe content."""
    user_input_lower = user_input.lower()
    for keyword in UNSAFE_KEYWORDS:
        if keyword in user_input_lower:
            return True
    return False

# ── 4. Send Query to Groq API ────────────────────────────────
# Exact models from your Groq account dropdown
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
    "groq/compound-mini",
    "openai/gpt-oss-safeguard-20b",
]

def get_health_response(user_query):
    """
    Send user query to Groq API.
    Auto tries multiple models until one works.
    """
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_query}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            if any(x in error_msg for x in
                   ["model_not_found", "decommissioned",
                    "404", "does not exist", "no access"]):
                continue
            elif any(x in error_msg for x in
                     ["authentication", "api_key", "401",
                      "invalid_api_key", "AuthenticationError"]):
                return "ERROR: Invalid API key. Please check your key at https://console.groq.com/keys"
            elif any(x in error_msg for x in ["rate_limit", "429"]):
                return "ERROR: Rate limit reached. Please wait and try again."
            else:
                return f"ERROR: {error_msg}"

    return "ERROR: No models worked. Please check your Groq account."

# ── 5. Main Chatbot Loop ──────────────────────────────────────
def run_chatbot():
    """Main chatbot loop."""
    print("\n" + "=" * 50)
    print("   GENERAL HEALTH QUERY CHATBOT")
    print("   DevelopersHub – AI/ML Internship Task 4")
    print("   Powered by Groq API (Free)")
    print("=" * 50)
    print("Ask any general health question.")
    print("Type 'quit' to exit.\n")

    print("Example questions you can ask:")
    print("  → What causes a sore throat?")
    print("  → Is paracetamol safe for children?")
    print("  → How can I lower my blood pressure naturally?")
    print("  → What are the symptoms of diabetes?")
    print("  → How many hours of sleep does an adult need?")
    print("-" * 50 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nChatbot: Thank you! Stay healthy!")
            break

        if not user_input:
            print("Chatbot: Please type a question.\n")
            continue

        if is_unsafe(user_input):
            print("\nChatbot: I'm sorry, I'm not able to help with that topic.")
            print("         If you are in distress, please contact a mental")
            print("         health professional or call a helpline.\n")
            continue

        print("\nChatbot: ", end="", flush=True)
        response = get_health_response(user_input)
        print(response)
        print()

# ── 6. Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    run_chatbot()