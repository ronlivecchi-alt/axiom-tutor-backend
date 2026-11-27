import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
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
gemini_client = None
# The new SDK doesn't need the "models/" prefix
GEMINI_MODEL = "gemini-3.8-flash" 

if gemini_key:
    try:
        # The new SDK uses genai.Client() instead of genai.configure()
        gemini_client = genai.Client(api_key=gemini_key)
        print(f"Gemini client initialized successfully (model: {gemini_client})")
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
        "gemini_ready": gemini_client is not None,
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
    if gemini_client is None:
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
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
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
    
@app.get("/api/metrics/{class_code}")
async def get_teacher_metrics(class_code: str):
    if not supabase:
        return {"total_submissions": 0, "class_accuracy": 0, "top_misconceptions": []}

    try:
        # 1. Get all students in this specific class
        student_res = supabase.table("students").select("id").eq("class_code", class_code).execute()
        student_ids = [str(s["id"]) for s in student_res.data] if student_res.data else []
        
        if not student_ids:
            return {"total_submissions": 0, "class_accuracy": 0, "top_misconceptions": []}

        # 2. Fetch submissions and filter for only these students
        response = supabase.table("student_submissions").select("*").execute()
        data = [row for row in (response.data or []) if row.get("student_id") in student_ids]

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
            "top_misconceptions": top_misconceptions
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

    if supabase and student_id and user_message:
        try:
            supabase.table("chat_logs").insert({
                "student_id": student_id,
                "sender": "Student",
                "message": user_message
            }).execute()
        except Exception as e:
            print(f"Log error: {e}")
    if not user_message:
        return {"reply": "Ask me anything about the problem we're working on!"}

    if gemini_client is None:
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
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        reply = response.text.strip()
        if supabase and student_id and reply:
            try:
                supabase.table("chat_logs").insert({
                    "student_id": student_id,
                    "sender": "Axiom",
                    "message": reply
                }).execute()
            except Exception as e:
                print(f"Log error: {e}")

        return {"reply": reply}
        
        if not reply:
            reply = "Hmm, let me think about that. Can you ask in a different way?"
            
        # Hard limit to 500 characters (about 3-4 sentences)
        if len(reply) > 500:
            reply = reply[:497] + "..."
            
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
    criteria_focus: str = "Auto" 

@app.post("/api/generate-problem")
async def generate_problem(req: GenerateRequest):
    if not supabase or gemini_client is None:
        return {"error": "Services not ready"}

    try:
        # 1. Fetch the specific topic from your Supabase curriculum table
        clean_topic = req.topic.strip() if req.topic else ""
        response = supabase.table("curriculum").select("*").eq("topic", clean_topic).execute()
        
        if not response.data:
            return {"error": "Topic not found in curriculum. Check your Supabase table."}
        
        curriculum_data = response.data[0]

        # 2. Safely get the data
        unit_name = curriculum_data.get('unit_name', 'MYP 1 Math')
        topic = curriculum_data.get('topic', req.topic)
        objectives = curriculum_data.get('objectives', 'Understand the core concepts.')
        example_problem = curriculum_data.get('example_problem', 'No example provided.')
        criteria_focus = req.criteria_focus if hasattr(req, 'criteria_focus') else "Auto"

        # 3. Build the prompt
        custom_content = curriculum_data.get('content', '')
        context_text = f"Here is the exact textbook content to base the problem on:\n{custom_content}" if custom_content else ""

        prompt = f"""You are an expert IB MYP 1 mathematics teacher.
I need a new practice problem for my students based on the following curriculum:

Unit: {unit_name}
Topic: {topic}
Learning Objectives: {objectives}
Example Problem: {example_problem}

{context_text}

CRITERIA INSTRUCTION:
The requested focus is: {criteria_focus}.
- If Criterion A: Generate a standard, single-part math problem.
- If Criterion B (Investigating Patterns): Generate a multi-part question (Part a, b, c) that guides the student to discover a pattern.
- If Criterion C (Communicating): The problem should ask the student to explain their reasoning.
- If Criterion D (Real-world): Apply the math to a real-world scenario.

CRITICAL RULES:
1. If generating a multi-part question (Criterion B), you MUST put the actual questions for the student to answer inside the "parts" array (e.g., ["Part a: Find the next two terms.", "Part b: Write the general rule.", "Part c: Find the 50th term."]).
2. Do NOT put the questions inside the "worked_example". The worked_example must ONLY contain the step-by-step solution to the problem.
3. The "description" should be a brief 1-sentence introduction to the problem.

Respond ONLY with valid JSON. Do not use markdown formatting, do not use code blocks, do not add any text before or after the JSON.
The JSON must have exactly these keys:
{{
  "question_title": "A short title",
  "equation": "The main math expression in LaTeX format",
  "description": "A short 1-sentence instruction for the student.",
  "correct_answer": "The final exact correct answer.",
  "worked_example": ["Step 1: ...", "Step 2: ...", "Answer: ..."],
  "criteria": "You MUST output exactly one of these strings: 'Criterion A', 'Criterion B', 'Criterion C', or 'Criterion D'. Do not output anything else in this field.",
  "parts": ["Part a: ...", "Part b: ...", "Part c: ..."] // If Criterion B, you MUST populate this array with the student questions. Do NOT put the questions in the description.",
}}
"""
        # 4. Call Gemini
        ai_response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
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

        # 1. Fetch the existing content first
        existing_data = supabase.table("curriculum").select("content").eq("topic", topic).execute()
        
        existing_text = ""
        if existing_data.data and existing_data.data[0].get("content"):
            existing_text = existing_data.data[0]["content"]

        # 2. Combine old text and new text with a divider
        # We add a little header so the AI knows this is a new document
        combined_text = existing_text + f"\n\n--- New Material: {file.filename} ---\n\n" + extracted_text

        # 3. Save the combined text (still keeping the 10,000 char limit)
        supabase.table("curriculum").update({
            "content": combined_text[:20000]
        }).eq("topic", topic).execute()

        return {"message": f"Successfully uploaded and processed {file.filename} for {topic}!"}
    except Exception as e:
        print(f"Upload error: {e}")
        return {"error": str(e)}
# --- NEW DYNAMIC CURRICULUM ROUTES ---

@app.get("/api/topics")
async def get_all_topics():
    if not supabase:
        return []
    try:
        # Fetch just the unit and topic names to build the dropdown
        response = supabase.table("curriculum").select("unit_name, topic, objectives").execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching topics: {e}")
        return []

class NewTopicRequest(BaseModel):
    unit_name: str
    topic: str
    objectives: str = ""
    example_problem: str = ""

@app.post("/api/topics")
async def create_new_topic(req: NewTopicRequest):
    if not supabase:
        return {"error": "Database not ready"}
    try:
        # Insert the new row
        supabase.table("curriculum").insert({
            "unit_name": req.unit_name,
            "topic": req.topic,
            "objectives": req.objectives,
            "example_problem": req.example_problem,
            "content": "" # Empty content to start
        }).execute()
        return {"message": "Topic added successfully!"}
    except Exception as e:
        print(f"Error adding topic: {e}")
        return {"error": str(e)}
class SaveProblemRequest(BaseModel):
    unit_name: str
    topic: str
    title: str
    equation: str
    description: str
    correct_answer: str
    worked_example: list
    criteria: str = "Criterion A"   
    parts: list = []

@app.post("/api/save-problem")
async def save_problem(req: SaveProblemRequest):
    if not supabase:
        return {"error": "Database not ready"}
    try:
        # Make sure this exact line with the opening bracket { is there!
        supabase.table("problems").insert({
            "unit_name": req.unit_name,
            "topic": req.topic,
            "title": req.title,
            "equation": req.equation,
            "description": req.description,
            "correct_answer": req.correct_answer,
            "worked_example": req.worked_example,
            "criteria": req.criteria,
            "parts": req.parts
        }).execute()
        
        return {"message": "Problem saved successfully!"}
    except Exception as e:
        print(f"Error saving problem: {e}")
        return {"error": str(e)}
@app.get("/api/problems/{topic}")
async def get_problems(topic: str):
    if not supabase:
        return []
    try:
        response = supabase.table("problems").select("*").eq("topic", topic).execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching problems: {e}")
        return []
import random

@app.get("/api/quiz/{topic}")
async def get_quiz(topic: str):
    if not supabase:
        return []
    try:
        # Add .strip() to handle any trailing spaces
        clean_topic = topic.strip() if topic else ""
        
        response = supabase.table("problems").select("*").eq("topic", clean_topic).execute()
        all_problems = response.data or []
        
        if len(all_problems) <= 5:
            import random
            random.shuffle(all_problems)
            return all_problems
            
        import random
        return random.sample(all_problems, 5)
    except Exception as e:
        print(f"Error fetching quiz: {e}")
        return []
class StudyToolsRequest(BaseModel):
    topic: str

@app.post("/api/study-tools")
async def generate_study_tools(req: StudyToolsRequest):
    if not supabase or gemini_client is None:
        return {"error": "Services not ready"}

    try:
        # Fetch the curriculum text for this topic
        response = supabase.table("curriculum").select("content", "objectives").eq("topic", req.topic).execute()
        if not response.data:
            return {"error": "No curriculum found for this topic."}
        
        data = response.data[0]
        content = data.get('content', '')
        objectives = data.get('objectives', '')
        
        if not content:
            return {"error": "No textbook content uploaded for this topic yet."}

        prompt = f"""You are an expert IB MYP math teacher. 
Based on the following curriculum content, create study materials for a student.

Learning Objectives: {objectives}
Textbook Content:
{content}

Generate a JSON response with EXACTLY this structure (do not use markdown blocks, just raw JSON):
{{
  "flashcards": [
    {{"front": "Term or Question", "back": "Definition or Answer"}},
    {{"front": "Term or Question", "back": "Definition or Answer"}}
  ],
  "graphic_organizer": [
    "Main Concept: ...",
    "Key Rule: ...",
    "Step-by-step: ...",
    "Common Mistake to Avoid: ..."
  ]
}}
Generate between 3 and 5 flashcards, and 3 to 5 bullet points for the graphic organizer.
"""
        ai_response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        text = ai_response.text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        import json
        tools = json.loads(text)
        return tools

    except Exception as e:
        print(f"Study Tools Error: {e}")
        return {"error": str(e)}
class StudentLogin(BaseModel):
    class_code: str
    name: str
    pin: str

@app.post("/api/login")
async def student_login(login_data: StudentLogin):
    if not supabase:
        return {"error": "Database not ready"}
    try:
        # Check if the student exists and the PIN matches
        response = supabase.table("students").select("*").eq("class_code", login_data.class_code).eq("name", login_data.name).eq("pin", login_data.pin).execute()
        
        if not response.data:
            return {"error": "Invalid Class Code, Name, or PIN"}
            
        student = response.data[0]
        return {"success": True, "student": {"id": student["id"], "name": student["name"], "class": student["class_code"]}}
    except Exception as e:
        print(f"Login error: {e}")
        return {"error": str(e)}

class TestRequest(BaseModel):
    topics: list[str]

@app.post("/api/generate-test")
async def generate_test(req: TestRequest):
    if not supabase:
        return []
    try:
        all_problems = []
        seen_equations = set() # <--- This prevents duplicates
        
        for topic in req.topics:
            clean_topic = topic.strip() if topic else ""
            response = supabase.table("problems").select("*").eq("topic", clean_topic).execute()
            if response.data:
                for prob in response.data:
                    eq = prob.get("equation", "")
                    # Only add if we haven't seen this exact equation yet
                    if eq not in seen_equations:
                        all_problems.append(prob)
                        seen_equations.add(eq)
        
        import random
        random.shuffle(all_problems)
        return all_problems[:10] # Return up to 10 unique questions
    except Exception as e:
        print(f"Error generating test: {e}")
        return [] 
        
@app.get("/api/students")
async def get_all_students():
    if not supabase:
        return []
    try:
        response = supabase.table("students").select("*").execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching students: {e}")
        return []

@app.get("/api/classes")
async def get_classes():
    if not supabase:
        return []
    try:
        response = supabase.table("students").select("class_code").execute()
        # Extract unique class codes, ignoring empty ones
        classes = list(set([row["class_code"] for row in response.data if row.get("class_code")]))
        return sorted(classes)
    except Exception as e:
        print(f"Error fetching classes: {e}")
        return []

@app.get("/api/chat-logs/{student_id}")
async def get_chat_logs(student_id: str):
    if not supabase:
        return []
    try:
        response = supabase.table("chat_logs").select("*").eq("student_id", student_id).order("created_at").execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching chat logs: {e}")
        return []
