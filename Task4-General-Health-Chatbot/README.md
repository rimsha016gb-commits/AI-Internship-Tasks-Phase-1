# Task 4: General Health Query Chatbot (Prompt Engineering)

**Internship:** DevelopersHub Corporation – AI/ML Engineering  


## Objective

Create a chatbot that answers general health-related questions
using an LLM (Large Language Model) via Groq API with
prompt engineering and safety filtering.

---

## Tools Used

| Tool | Details |
|------|---------|
| Groq API | Free LLM API — https://console.groq.com |
| Model | openai/gpt-oss-120b (with auto fallback) |
| Technique | Prompt Engineering + Safety Filtering |
| Language | Python 3.10 |

---

## What is Prompt Engineering?

Prompt Engineering means writing smart instructions to control
how the AI model responds. In this chatbot, we use a system
prompt to make the model behave like a helpful medical assistant:

```
"You are a friendly and knowledgeable health assistant.
 NEVER diagnose or prescribe medication.
 Always recommend consulting a doctor..."
```

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| groq | Connect to Groq API and use LLM models |

---

## Install Requirements

```
pip install -r requirements.txt
```

---

## Project Structure

```
Task4/
├── task4_health_chatbot.py      # Main Python script
├── Task4_Health_Chatbot.ipynb   # Jupyter Notebook
├── requirements.txt              # Required libraries
└── README.md                     # Project documentation
```

---

## How to Get Free Groq API Key

1. Go to https://console.groq.com
2. Sign up with Google account
3. Click "API Keys" in left sidebar
4. Click "Create API Key"
5. Copy your key (starts with gsk_...)

---

## How to Run

### Step 1 — Add API Key (Line 12 in .py file)
```python
client = Groq(api_key="gsk_your-actual-key-here")
```

### Step 2 — Install requirements
```
pip install -r requirements.txt
```

### Step 3 — Run Python Script
```
python task4_health_chatbot.py
```

### Option — Run Jupyter Notebook
```
jupyter notebook Task4_Health_Chatbot.ipynb
```

---

## Models Used (Auto Fallback)

| Priority | Model | Status |
|----------|-------|--------|
| 1st | openai/gpt-oss-120b | ✅ Primary |
| 2nd | openai/gpt-oss-20b | ✅ Backup |
| 3rd | groq/compound | ✅ Backup |
| 4th | groq/compound-mini | ✅ Backup |

---

## Safety Filter

The chatbot automatically blocks these types of queries:

| Blocked Topic | Response |
|---------------|---------|
| Self-harm | Redirects to helpline |
| Suicide | Redirects to helpline |
| Drug abuse | Blocked with message |
| Illegal drugs | Blocked with message |

---

## Example Queries and Responses

**Query:** What causes a sore throat?  
**Response:** A sore throat is most commonly caused by viral
infections such as cold or flu...
Please consult a doctor for personalized medical advice.

**Query:** Is paracetamol safe for children?  
**Response:** Paracetamol can be used for children but the
dosage depends on the child's weight and age...
Please consult a doctor for personalized medical advice.

---

## Key Results and Findings

1. Prompt engineering successfully controls chatbot tone and behavior.
2. System prompt ensures responses are always friendly and safe.
3. Safety filter correctly blocks all harmful queries.
4. Groq API provides free and fast LLM responses.
5. Auto model fallback ensures chatbot always works.
6. Chatbot never diagnoses or prescribes medication.

---

## Conclusion

The Health Query Chatbot successfully answers general health
questions using prompt engineering with Groq API. The system
prompt defines clear boundaries — the chatbot is always
friendly, never diagnoses, always recommends doctor
consultation, and blocks unsafe queries automatically.

---

## Author

**Name:** Rimsha Aslam 
**Internship:** AI/ML Engineering Intern  
**Organization:** DevelopersHub Corporation

