import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai
import io
from fastapi import UploadFile, File, Form
from pypdf import PdfReader

# 1. Initialize FastAPI app FIRST
app = FastAPI()

# 2. Enable CORS so the Netlify frontend can call this backend later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize Supabase
url: str = os.environ.get("SUPABASE_URL", "")
key: str = (
    os.environ.get("SUPABASE_SECRET_KEY")
    or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_ANON_KEY", "")
)

supabase: Client = None
if url and key:
    try:
        supabase = create_client(url, key)
        print("Supabase client initialized successfully")
    except Exception as e:
        print(f"Supabase init error: {e}")
else:
    print(f"Supabase NOT initialized. url set: {bool(url)}, key set: {bool(key)}")

# 4. Initialize Gemini
gemini_key: str = os.environ.get("GEMINI_API_KEY", "")
gemini_model = None
GEMINI_MODEL = "models/gemini-3.8-flash"
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        # Try a current model; if it fails we'll fall back
        gemini_model = genai.GenerativeModel("models/gemini-3.8-flash")
        print(f"Gemini client initialized successfully (model: {GEMINI_MODEL})")
    except Exception as e:
        print(f"Gemini init error: {e}")
else:
    print(f"Gemini NOT initialized. GEMINI_API_KEY set: {bool(gemini_key)}")

# Data Models
class StudentSubmission(BaseModel):
    student_id: str
    question_id: str
    user_answer: str
    problem_text: str = ""      # optional, e.g. "2/3 + 1/4 = ?"
    correct_answer: str = ""    # optional, e.g. "11/12"

# Routes
@app.get("/api")
@app.get("/api/")
def api_root():
    return {"message": "Axiom AI Tutor API is live!"}

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "Axiom AI Tutor API",
        "supabase_ready": supabase is not None,
        "gemini_ready": gemini_model is not None,
    }

# Fallback rule-based evaluation (used if Gemini fails)
def fallback_evaluate(user_answer: str, correct_answer: str):
    user_clean = user_answer.lower().strip()
    correct_clean = correct_answer.lower().strip()
    if correct_clean and correct_clean in user_clean:
        return True, "Great job! That is correct.", None
    return False, "Not quite — try again, or ask for a hint.", None

# Gemini-powered evaluation
def evaluate_with_gemini(user_answer: str, problem_text: str, correct_answer: str):
    if gemini_model is None:
        return None

    prompt = f"""You are a Socratic math tutor for IB MYP 1 students (age 11-12).

A student was given this problem:
{problem_text if problem_text else "(problem text not provided)"}

The correct answer is:
{correct_answer if correct_answer else "(not provided — use your judgment)"}

The student answered:
{user_answer}

Evaluate the student's answer. Respond ONLY with valid JSON in exactly this format, no other text:
{{
  "is_correct": true or false,
  "misconception_tag": "a short 2-5 word tag describing the type of error, or null if correct",
  "feedback": "a short, encouraging message for the student (max 2 sentences, Grade 6 friendly)"
}}

Rules:
- is_correct is true if the answer is mathematically equivalent to the correct answer, regardless of formatting.
- If the student is wrong, misconception_tag should describe WHY they got it wrong (e.g. "Added denominators directly", "Sign error", "Confused factors and multiples").
- Be encouraging even when wrong. Never say "you failed."
- Do not include markdown code fences in your response.
"""

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()

        # Strip any accidental markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        return {
            "is_correct": bool(data.get("is_correct", False)),
            "misconception_tag": data.get("misconception_tag"),
            "feedback": data.get("feedback", "Let's keep working on this."),
        }
    except Exception as e:
        print(f"Gemini evaluation error: {e}")
        return None

@app.post("/api/evaluate")
async def evaluate_submission(submission: StudentSubmission):
    # Try Gemini first
    result = evaluate_with_gemini(
        submission.user_answer,
        submission.problem_text,
        submission.correct_answer,
    )

    # Fall back to rule-based if Gemini failed or isn't configured
    if result is None:
        is_correct, feedback, misconception_tag = fallback_evaluate(
            submission.user_answer, submission.correct_answer
        )
    else:
        is_correct = result["is_correct"]
        feedback = result["feedback"]
        misconception_tag = result["misconception_tag"]

    # Store in Supabase
    if supabase:
        try:
            supabase.table("student_submissions").insert({
                "student_id": submission.student_id,
                "question_id": submission.question_id,
                "user_answer": submission.user_answer,
                "is_correct": is_correct,
                "misconception_tag": misconception_tag,
            }).execute()
        except Exception as e:
            print(f"DB Insert Error: {e}")

    return {
        "is_correct": is_correct,
        "feedback": feedback,
        "misconception_tag": misconception_tag,
    }

@app.get("/api/metrics")
async def get_teacher_metrics():
    if not supabase:
        return {"total_submissions": 0, "class_accuracy": 0, "top_misconceptions": []}

    try:
        response = supabase.table("student_submissions").select("*").execute()
        data = response.data or []

        total = len(data)
        if total == 0:
            return {"total_submissions": 0, "class_accuracy": 0, "top_misconceptions": []}

        correct_count = sum(1 for row in data if row.get("is_correct"))
        accuracy = round((correct_count / total) * 100, 1)

        misconception_counts = {}
        for row in data:
            tag = row.get("misconception_tag")
            if tag:
                misconception_counts[tag] = misconception_counts.get(tag, 0) + 1

        top_misconceptions = [
            {"tag": tag, "count": count}
            for tag, count in sorted(misconception_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        return {
            "total_submissions": total,
            "class_accuracy": accuracy,
            "top_misconceptions": top_misconceptions,
        }
    except Exception as e:
        print(f"Metrics Fetch Error: {e}")
        return {"total_submissions": 0, "class_accuracy": 0, "top_misconceptions": []}
@app.post("/api/chat")
async def chat_with_tutor(message: dict):
    """
    Socratic chat endpoint.
    Expects JSON: { "message": str, "student_id": str, "problem_key": str, "problem_text": str }
    Returns: { "reply": str }
    """
    user_message = message.get("message", "").strip()
    problem_text = message.get("problem_text", "").strip()
    problem_key = message.get("problem_key", "").strip()
    student_id = message.get("student_id", "").strip()

    if not user_message:
        return {"reply": "Ask me anything about the problem we're working on!"}

    if gemini_model is None:
        return {"reply": "I'm having trouble thinking right now. Try again in a moment!"}

    system_prompt = """You are Axiom, a Socratic math tutor for IB MYP 1 students (age 11-12).

Your job is to help students THINK through problems, not to give them the answer.

CRITICAL RULES:
- Keep every response to ONE or TWO short sentences. Nothing longer.
- If the student asks for the answer directly, reply in ONE sentence that gently refuses and asks a guiding question.
- Use simple Grade 6 language.
- Reference specific numbers from the problem.
- Never say "wrong" or "you failed."
- Do not write paragraphs. Do not explain fully. Just nudge the student forward.

You are a coach, not an answer key. Brevity is essential."""

    problem_context = f"\n\nThe student is currently working on: {problem_text}" if problem_text else ""
    full_prompt = f"{system_prompt}{problem_context}\n\nStudent: {user_message}\n\nAxiom:"

    try:
        response = gemini_model.generate_content(full_prompt)
        reply = response.text.strip()
        if not reply:
            reply = "Hmm, let me think about that. Can you ask in a different way?"
        # Hard limit to 100 characters to survive proxy truncation
        if len(reply) > 100:
            reply = reply[:97] + "..."
        print(f"Chat reply length: {len(reply)}")
        return {"reply": reply}        
    except Exception as e:
        print(f"Chat error: {e}")
    return {"reply": "I had trouble thinking about that. Try asking again in a different way!"}
    
@app.get("/api/roster")
async def get_class_roster():
    if not supabase:
        return []
        
    try:
        # Fetch all submissions
        response = supabase.table("student_submissions").select("*").execute()
        data = response.data or []

        # Group the data by student
        roster_stats = {}
        for row in data:
            sid = row.get("student_id")
            if not sid:
                continue
                
            if sid not in roster_stats:
                roster_stats[sid] = {
                    "student_id": sid,
                    "total_attempts": 0,
                    "correct_attempts": 0,
                    "latest_misconception": None
                }
            
            roster_stats[sid]["total_attempts"] += 1
            if row.get("is_correct"):
                roster_stats[sid]["correct_attempts"] += 1
            elif row.get("misconception_tag"):
                # Keep track of the most recent misconception
                roster_stats[sid]["latest_misconception"] = row.get("misconception_tag")

        # Convert dictionary to a list for the frontend
        return list(roster_stats.values())
        
    except Exception as e:
        print(f"Roster Fetch Error: {e}")
        return []
class GenerateRequest(BaseModel):
    topic: str

@app.post("/api/generate-problem")
async def generate_problem(req: GenerateRequest):
    if not supabase or gemini_model is None:
        return {"error": "Services not ready"}

    try:
        # 1. Fetch the specific topic from your Supabase curriculum table
        response = supabase.table("curriculum").select("*").eq("topic", req.topic).execute()
        if not response.data:
            return {"error": "Topic not found in curriculum. Check your Supabase table."}
        
        curriculum_data = response.data[0]
        
        # 2. Safely get the data (prevents crashes if a column is missing)
        unit_name = curriculum_data.get('unit_name', 'MYP 1 Math')
        topic = curriculum_data.get('topic', req.topic)
        objectives = curriculum_data.get('objectives', 'Understand the core concepts.')
        example_problem = curriculum_data.get('example_problem', 'No example provided.')
        
        # 3. Build the prompt
        # Grab the custom textbook/slides text if it exists
        custom_content = curriculum_data.get('content', '')
        context_text = f"Here is the exact textbook content to base the problem on:\n{custom_content}" if custom_content else ""
                # 3. Build the prompt
        prompt = f"""You are an expert IB MYP 1 mathematics teacher.
I need a new practice problem for my students based on the following curriculum:

Unit: {unit_name}
Topic: {topic}
Learning Objectives: {objectives}
Example Problem: {example_problem}

Here is the exact textbook content to base the problem on:
{context_text}

Note: The textbook text above may have been extracted from a PDF and might have messy formatting, missing symbols, or strange spacing for math equations. Please use your best judgment to interpret the mathematical meaning and output clean, properly formatted LaTeX.

Generate ONE new, unique practice problem for this topic. 
Respond ONLY with valid JSON. Do not use markdown formatting, do not use code blocks, do not add any text before or after the JSON.
The JSON must have exactly these keys:
{{
  "question_title": "A short title",
  "equation": "The math expression in LaTeX format. Use double backslashes for LaTeX commands (e.g., '\\\\frac{2}{5} + \\\\frac{1}{3} = ?')",
  "description": "A short hint or instruction for the student.",
  "correct_answer": "The exact correct answer (e.g., '11/15')",
  "worked_example": ["Step 1: ...", "Step 2: ...", "Answer: ..."]
}}
"""
        # 4. Call Gemini
        ai_response = gemini_model.generate_content(prompt)
        text = ai_response.text.strip()
        
        # 5. Robustly strip markdown blocks if Gemini ignores instructions
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        # 6. Parse the JSON safely
        import json
        try:
            new_problem = json.loads(text)
        except Exception as json_err:
            print(f"RAW GEMINI TEXT: {text}")
            raise Exception(f"JSON parsing failed. Gemini output: {text[:100]}...")
            
        return new_problem

    except Exception as e:
        print(f"Generation error: {e}")
        return {"error": str(e)}

    except Exception as e:
        print(f"Generation error: {e}")
        return {"error": "Failed to generate problem"}

@app.post("/api/upload-material")
async def upload_material(topic: str = Form(...), file: UploadFile = File(...)):
    if not supabase:
        return {"error": "Database not ready"}
    
    try:
        # Read the file
        contents = await file.read()
        
        # Extract text based on file type
        extracted_text = ""
        if file.filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(contents))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        elif file.filename.endswith(".txt"):
            extracted_text = contents.decode("utf-8")
        else:
            return {"error": "Please upload a PDF or TXT file for now."}

        # Debug: Print how many characters were extracted so we can see it in Render logs
        print(f"Extracted {len(extracted_text)} characters from {file.filename}")

        # Save to Supabase
        # Note: We limit it to ~10,000 characters to prevent overloading the AI context window later
        supabase.table("curriculum").update({
            "content": extracted_text[:10000]
        }).eq("topic", topic).execute()

        return {"message": f"Successfully uploaded and processed {file.filename} for {topic}!"}
    except Exception as e:
        print(f"Upload error: {e}")
        return {"error": str(e)}
