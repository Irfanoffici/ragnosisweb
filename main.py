from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import uvicorn
import time
import requests
import json
import re
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import hashlib

# Load environment variables
load_dotenv()

# Security
security = HTTPBearer()
API_TOKENS = set()

# Data models
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[dict]] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    response_time: float
    model_used: str
    follow_up_questions: List[str]
    session_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    groq_status: str
    gemini_status: str
    wikipedia_status: str
    local_db_status: str

# Global variables - initialize as None
groq_client = None
gemini_client = None
wiki_wiki = None
system_startup_time = datetime.now()
query_counter = 0
local_medical_data = {}
user_sessions = {}

print("🤖 Starting Ragnosis AI - Advanced Medical Assistant...")

def initialize_secure_tokens():
    """Initialize API tokens"""
    global API_TOKENS
    default_token = os.getenv("CLARA_API_TOKEN", "ragnosis-ai-token-2024")
    API_TOKENS.add(default_token)
    print(f"🔑 API Tokens initialized: {len(API_TOKENS)} tokens loaded")

def initialize_groq_client():
    """Initialize Groq client with dynamic import"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found")
        return None
    
    try:
        import groq
        client = groq.Groq(api_key=groq_api_key)
        # Test connection
        test_response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'ready'"}],
            model="llama-3.1-8b-instant",
            max_tokens=5
        )
        print("✅ Groq client initialized successfully!")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize Groq client: {e}")
        return None

def initialize_gemini_client():
    """Initialize Gemini client with dynamic import and proper model"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found")
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        
        # List available models and use the first available one
        models = genai.list_models()
        available_models = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
        
        if available_models:
            # Use the first available model
            model_name = available_models[0]
            model = genai.GenerativeModel(model_name)
            # Test connection
            response = model.generate_content("Hello")
            print(f"✅ Gemini client initialized successfully with model: {model_name}")
            return model
        else:
            print("❌ No suitable Gemini models found")
            return None
            
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        return None

def initialize_wikipedia():
    """Initialize Wikipedia API with dynamic import"""
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="RagnosisAI/1.0"
        )
        print("✅ Wikipedia API initialized successfully!")
        return wiki
    except Exception as e:
        print(f"❌ Failed to initialize Wikipedia: {e}")
        return None

def initialize_local_medical_database():
    """Initialize and load local medical database"""
    medical_data = {}
    try:
        # Try to load from medical_data.txt
        with open('data/medical_data.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse medical data
        sections = content.split('\n\n')
        for section in sections:
            lines = section.split('\n')
            if len(lines) >= 2:
                key = lines[0].lower().replace(':', '').strip()
                value = '\n'.join(lines[1:]).strip()
                medical_data[key] = value
        
        print(f"✅ Local medical database loaded: {len(medical_data)} conditions")
        return medical_data
    except Exception as e:
        print(f"❌ Failed to load local medical database: {e}")
        # Create default medical data
        default_data = {
            'headache': 'Common causes: tension, dehydration, sinus issues\nTreatment: rest, hydration, pain relievers\nWhen to seek help: severe pain, vision changes',
            'fever': 'Definition: body temperature above 100.4°F\nManagement: rest, hydration, fever-reducers\nEmergency: fever over 103°F, lasting more than 3 days',
            'cough': 'Types: dry cough, productive cough\nRemedies: honey, humidifier, hydration\nSee doctor if: lasts over 3 weeks',
            'cold': 'Symptoms: runny nose, sore throat, cough\nTreatment: rest, fluids, OTC medications\nPrevention: hand washing',
            'allergies': 'Triggers: pollen, dust, pet dander\nSymptoms: sneezing, itchy eyes, runny nose\nManagement: antihistamines, allergen avoidance',
            'pain': 'Types: acute, chronic\nManagement: rest, medication, physical therapy\nEmergency: severe pain, chest pain, head injury',
            'nausea': 'Causes: stomach virus, food poisoning, migraine\nRemedies: ginger, small meals, hydration\nSee doctor if: persistent, with fever, dehydration'
        }
        print("✅ Using default medical database")
        return default_data

def initialize_user_database():
    """Initialize SQLite database for user data and logs"""
    try:
        conn = sqlite3.connect('ragnosis_users.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # Create user sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                last_activity TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                user_data TEXT DEFAULT '{}'
            )
        ''')
        
        # Create conversation logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP,
                user_message TEXT,
                bot_response TEXT,
                model_used TEXT,
                response_time REAL,
                FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ User database initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize user database: {e}")
        return False

def search_local_medical_data(query: str, medical_data: dict):
    """Search local medical database for relevant information"""
    query_lower = query.lower()
    relevant_data = []
    
    # Simple keyword matching
    for condition, info in medical_data.items():
        if any(keyword in query_lower for keyword in condition.split()):
            relevant_data.append({
                'condition': condition,
                'information': info,
                'relevance_score': sum(1 for keyword in condition.split() if keyword in query_lower)
            })
    
    # Sort by relevance and return top 3
    relevant_data.sort(key=lambda x: x['relevance_score'], reverse=True)
    return relevant_data[:3]

def generate_ai_response_with_groq(messages: List[dict]):
    """Generate response using Groq"""
    global groq_client
    if not groq_client:
        raise Exception("Groq client not available")
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        raise Exception("Groq service unavailable")

def generate_ai_response_with_gemini(prompt: str):
    """Generate response using Gemini"""
    global gemini_client
    if not gemini_client:
        raise Exception("Gemini client not available")
    
    try:
        response = gemini_client.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        raise Exception("Gemini service unavailable")

def generate_response_with_local_data(query: str, conversation_history: List[dict], medical_data: dict, session_id: str):
    """Generate response using local medical database"""
    
    # Search local medical database
    local_results = search_local_medical_data(query, medical_data)
    
    # Build context from local data
    local_context = ""
    if local_results:
        local_context = "\n📚 LOCAL MEDICAL KNOWLEDGE BASE:\n"
        for result in local_results:
            local_context += f"• {result['condition'].title()}: {result['information'][:200]}...\n"
    
    # Enhanced local response generation
    if local_results:
        best_match = local_results[0]
        response = f"**Based on my medical knowledge base for {best_match['condition'].title()}:**\n\n"
        response += f"{best_match['information']}\n\n"
        
        # Add personalized follow-up
        if 'pain' in query.lower() or 'hurt' in query.lower():
            response += "To help you better, I'd like to know:\n• How severe is the pain (scale 1-10)?\n• How long have you been experiencing this?"
        else:
            response += "To provide more specific advice, could you tell me:\n• How long have you had these symptoms?\n• Any other symptoms you're experiencing?"
        
    else:
        response = f"**Regarding your question about '{query}':**\n\n"
        response += "I've checked my medical knowledge base, but I don't have specific information about this condition. "
        response += "For accurate medical advice, I recommend consulting with a healthcare professional.\n\n"
        response += "In the meantime, I'd be happy to help with general health information or answer other questions you might have."
    
    response += "\n\n💡 Remember: I'm an AI assistant for informational purposes. Always consult healthcare professionals for medical advice, diagnoses, or treatment."
    
    follow_up_questions = [
        "How long have you been experiencing this?",
        "Have you consulted a doctor about this?",
        "Are you taking any medications currently?",
        "Any other symptoms you're noticing?"
    ]
    
    return response, follow_up_questions

def log_conversation(session_id: str, user_message: str, bot_response: str, model_used: str, response_time: float):
    """Log conversation to database"""
    try:
        conn = sqlite3.connect('ragnosis_users.db', check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversation_logs 
            (session_id, timestamp, user_message, bot_response, model_used, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, datetime.now(), user_message, bot_response, model_used, response_time))
        
        # Update session activity
        cursor.execute('''
            UPDATE user_sessions 
            SET last_activity = ?, message_count = message_count + 1 
            WHERE session_id = ?
        ''', (datetime.now(), session_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Failed to log conversation: {e}")
        return False

def get_or_create_session(session_id: str = None):
    """Get existing session or create new one in database"""
    try:
        conn = sqlite3.connect('ragnosis_users.db', check_same_thread=False)
        cursor = conn.cursor()
        
        if not session_id:
            # Create new session
            new_session_id = f"session_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            cursor.execute('''
                INSERT INTO user_sessions (session_id, created_at, last_activity, message_count, user_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (new_session_id, datetime.now(), datetime.now(), 0, '{}'))
            conn.commit()
            conn.close()
            return new_session_id
        
        # Check if session exists
        cursor.execute('SELECT session_id FROM user_sessions WHERE session_id = ?', (session_id,))
        if cursor.fetchone():
            conn.close()
            return session_id
        else:
            # Create session with provided ID
            cursor.execute('''
                INSERT INTO user_sessions (session_id, created_at, last_activity, message_count, user_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, datetime.now(), datetime.now(), 0, '{}'))
            conn.commit()
            conn.close()
            return session_id
            
    except Exception as e:
        print(f"❌ Session management error: {e}")
        # Fallback to in-memory sessions
        if not session_id or session_id not in user_sessions:
            new_session_id = f"session_{int(time.time())}_{len(user_sessions)}"
            user_sessions[new_session_id] = {
                'created_at': datetime.now(),
                'conversation_history': [],
                'message_count': 0
            }
            return new_session_id
        return session_id

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup and shutdown events"""
    global groq_client, gemini_client, wiki_wiki, local_medical_data
    
    print("🔧 Initializing Ragnosis AI System...")
    
    initialize_secure_tokens()
    groq_client = initialize_groq_client()
    gemini_client = initialize_gemini_client()
    wiki_wiki = initialize_wikipedia()
    local_medical_data = initialize_local_medical_database()
    initialize_user_database()
    
    # System status report
    active_services = []
    if groq_client: active_services.append("Groq AI")
    if gemini_client: active_services.append("Gemini AI") 
    if wiki_wiki: active_services.append("Wikipedia")
    if local_medical_data: active_services.append("Local Medical DB")
    
    print(f"🎉 Ragnosis AI Ready! Active services: {', '.join(active_services)}")
    print("💬 Multi-layer fallback system: Groq → Gemini → Wikipedia → Local DB")
    print("📊 User data collection and logging enabled")
    
    yield
    
    print("🔧 Shutting down Ragnosis AI System...")

app = FastAPI(
    title="Ragnosis AI - Advanced Medical Assistant",
    description="Your friendly AI companion for medical information and health guidance",
    version="5.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def chat_interface():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Ragnosis AI - Medical Assistant</h1><p>Frontend files not found. API is running.</p>")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=str(datetime.now()),
        version="5.0.0",
        groq_status="connected" if groq_client else "disconnected",
        gemini_status="connected" if gemini_client else "disconnected",
        wikipedia_status="connected" if wiki_wiki else "disconnected",
        local_db_status="connected" if local_medical_data else "disconnected"
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    global query_counter
    start_time = time.time()
    query_counter += 1
    
    print(f"💬 Question received: {request.question}")
    
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        
        # Multi-layer response generation
        answer = ""
        model_used = ""
        follow_up_questions = []
        sources = []
        
        # Layer 1: Try Groq AI
        if groq_client and not answer:
            try:
                messages = [
                    {"role": "system", "content": "You are a medical AI assistant. Provide helpful, accurate information with a disclaimer about consulting healthcare professionals."},
                    {"role": "user", "content": request.question}
                ]
                answer = generate_ai_response_with_groq(messages)
                model_used = "groq-llama"
                follow_up_questions = ["Can you tell me more about your symptoms?", "How long have you experienced this?"]
            except Exception as e:
                print(f"❌ Groq failed: {e}")
        
        # Layer 2: Try Gemini AI
        if not answer and gemini_client:
            try:
                prompt = f"As a medical AI assistant, provide helpful information about: {request.question}. Include a disclaimer about consulting healthcare professionals."
                answer = generate_ai_response_with_gemini(prompt)
                model_used = "gemini-pro"
                follow_up_questions = ["Could you describe your symptoms in more detail?", "Have you seen a doctor about this?"]
            except Exception as e:
                print(f"❌ Gemini failed: {e}")
        
        # Layer 3: Use Local Medical Database (ALWAYS WORKS)
        if not answer:
            model_used = "local-medical-db"
            answer, follow_up_questions = generate_response_with_local_data(
                request.question, 
                request.conversation_history or [],
                local_medical_data,
                session_id
            )
            sources = ["📚 Ragnosis Local Medical Knowledge Base"]
        
        response_time = round(time.time() - start_time, 2)
        
        # Log conversation
        log_conversation(session_id, request.question, answer, model_used, response_time)
        
        print(f"✅ Response generated using {model_used} in {response_time}s")
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            response_time=response_time,
            model_used=model_used,
            follow_up_questions=follow_up_questions,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        # Ultimate fallback
        fallback_response = "I'm here to help! While I'm experiencing technical difficulties, please consult a healthcare professional for immediate medical concerns. For non-urgent matters, I'll be back to full functionality shortly."
        
        return QuestionResponse(
            answer=fallback_response,
            sources=[],
            response_time=round(time.time() - start_time, 2),
            model_used="emergency-fallback",
            follow_up_questions=["How can I help you today?", "Are you experiencing any specific symptoms?"],
            session_id=request.session_id or f"emergency_{int(time.time())}"
        )

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        conn = sqlite3.connect('ragnosis_users.db', check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM user_sessions')
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM conversation_logs')
        total_messages = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_queries": query_counter,
            "system_uptime": str(datetime.now() - system_startup_time)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info"
    )
