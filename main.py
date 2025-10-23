# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import groq
import os
import uvicorn
import time
import requests
import wikipediaapi
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import json
import re

# Load environment variables
load_dotenv()

# Security - Define this FIRST
security = HTTPBearer()
API_TOKENS = set()

# Enhanced Data models
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
    wikipedia_status: str

class SystemStatus(BaseModel):
    groq_connected: bool
    wikipedia_connected: bool
    startup_time: str
    total_queries_processed: int

# Global variables
embedding_model = None
collection = None
groq_client = None
wiki_wiki = None
system_startup_time = datetime.now()
query_counter = 0

# Session management
user_sessions = {}

print("🤖 Starting Ragnosis AI - Advanced Medical Assistant...")

def initialize_secure_tokens():
    """Initialize or load API tokens"""
    global API_TOKENS
    default_token = os.getenv("CLARA_API_TOKEN", "ragnosis-ai-token-2024")
    API_TOKENS.add(default_token)
    print(f"🔑 API Tokens initialized: {len(API_TOKENS)} tokens loaded")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    if token not in API_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token. Please check your authorization header."
        )
    return token

def initialize_groq_client():
    """Safely initialize Groq client with error handling"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment variables!")
        return None
    
    if groq_api_key.startswith("gsk_") and len(groq_api_key) > 30:
        try:
            client = groq.Groq(api_key=groq_api_key)
            # Test connection
            test_response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Say 'Ragnosis AI is ready to help!'"}],
                model="llama-3.1-8b-instant",
                max_tokens=15
            )
            print("✅ Groq client initialized successfully!")
            return client
        except Exception as e:
            print(f"❌ Failed to initialize Groq client: {e}")
            return None
    else:
        print("❌ Invalid Groq API key format")
        return None

def initialize_wikipedia():
    """Initialize Wikipedia API"""
    try:
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

def search_wikipedia_medical(query: str, max_results: int = 3):
    """Search Wikipedia for medical information"""
    if not wiki_wiki:
        return []
    
    try:
        search_results = []
        
        # More specific medical search terms
        medical_terms = [
            query,
            f"{query} (medicine)",
            f"{query} disease",
            f"{query} symptoms",
            f"{query} treatment",
            f"{query} diagnosis",
            f"{query} causes"
        ]
        
        for term in medical_terms:
            if len(search_results) >= max_results:
                break
                
            page = wiki_wiki.page(term)
            if page.exists() and len(page.summary) > 100:
                # Extract more relevant content
                preview = page.summary[:300] + "..." if len(page.summary) > 300 else page.summary
                search_results.append({
                    "title": page.title,
                    "url": page.fullurl,
                    "preview": preview,
                    "full_summary": page.summary[:800]  # More context for AI
                })
        
        return search_results[:max_results]
        
    except Exception as e:
        print(f"❌ Wikipedia search error: {e}")
        return []

def generate_ai_response(messages: List[dict], model: str = "llama-3.1-8b-instant", temperature: float = 0.7, max_tokens: int = 1000):
    """Generate response using Groq with enhanced AI personality"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return "😅 Oops! I'm having a little technical difficulty. Please try again in a moment!"

def generate_conversational_response(question: str, wikipedia_results: List[dict] = None, conversation_history: List[dict] = None, session_id: str = None):
    """Generate intelligent, conversational medical response with context"""
    
    # Build context from Wikipedia results
    wiki_context = ""
    if wikipedia_results:
        wiki_context = "\n\n📚 RELEVANT MEDICAL INFORMATION FROM WIKIPEDIA:\n"
        for i, result in enumerate(wikipedia_results, 1):
            wiki_context += f"{i}. {result['title']}: {result['preview']}\n"
    
    # Build conversation history context
    history_context = ""
    if conversation_history and len(conversation_history) > 0:
        history_context = "\n📝 OUR RECENT CONVERSATION (for context):\n"
        for msg in conversation_history[-6:]:  # Last 6 exchanges for better context
            if msg.get('role') == 'user':
                history_context += f"👤 User: {msg.get('content', '')}\n"
            else:
                history_context += f"🤖 Assistant: {msg.get('content', '')}\n"
    
    # Enhanced system prompt for better medical responses
    system_prompt = f"""You are Ragnosis AI, a friendly, knowledgeable, and compassionate medical AI assistant. 

PERSONALITY:
- Warm and approachable like a trusted healthcare friend
- Professional but conversational and easy to understand
- Empathetic and supportive
- Clear and specific in explanations
- Honest about limitations

CONTEXT:
{wiki_context}
{history_context}

USER'S CURRENT QUESTION: {question}

RESPONSE GUIDELINES:
1. BE SPECIFIC & ACCURATE: Provide detailed, medically accurate information
2. BE CONVERSATIONAL: Use natural language, appropriate emojis, and friendly tone
3. MAINTAIN CONTEXT: Reference previous conversation when relevant
4. BE PRACTICAL: Offer actionable advice when appropriate
5. USE EVIDENCE: Reference Wikipedia information when relevant
6. SAFETY FIRST: Always include a clear medical disclaimer
7. BE ENGAGING: Ask relevant follow-up questions to continue the conversation

MEDICAL DISCLAIMER (include at end):
"💡 Remember: I'm an AI assistant for informational purposes. Always consult healthcare professionals for medical advice, diagnoses, or treatment."

IMPORTANT: Your response should be comprehensive yet easy to understand. Break down complex medical concepts."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    return generate_ai_response(messages, temperature=0.7, max_tokens=1200)

def extract_follow_up_questions(response: str, current_question: str):
    """Extract relevant follow-up questions based on context"""
    # Pre-defined follow-up questions for common medical topics
    medical_follow_ups = {
        'symptoms': [
            "How long have you been experiencing these symptoms?",
            "Have you noticed any triggers that make it better or worse?"
        ],
        'diagnosis': [
            "Have you consulted a doctor about this?",
            "What tests or examinations have you had so far?"
        ],
        'treatment': [
            "What treatments have you tried already?",
            "Are you currently taking any medications?"
        ],
        'general': [
            "Would you like me to explain any part in more detail?",
            "Is there anything else you'd like to know about this?"
        ]
    }
    
    # Analyze the current question to choose relevant follow-ups
    question_lower = current_question.lower()
    
    if any(word in question_lower for word in ['symptom', 'feel', 'pain', 'hurt']):
        return medical_follow_ups['symptoms']
    elif any(word in question_lower for word in ['diagnos', 'test', 'result']):
        return medical_follow_ups['diagnosis']
    elif any(word in question_lower for word in ['treat', 'medic', 'therapy']):
        return medical_follow_ups['treatment']
    else:
        return medical_follow_ups['general']

def get_wikipedia_source_links(wikipedia_results: List[dict]):
    """Format Wikipedia results as source links"""
    sources = []
    for result in wikipedia_results:
        sources.append(f"📖 {result['title']} - {result['url']}")
    return sources

def get_or_create_session(session_id: str = None):
    """Get existing session or create new one"""
    if not session_id or session_id not in user_sessions:
        new_session_id = f"session_{int(time.time())}_{len(user_sessions)}"
        user_sessions[new_session_id] = {
            'created_at': datetime.now(),
            'conversation_history': [],
            'message_count': 0
        }
        return new_session_id, user_sessions[new_session_id]
    
    return session_id, user_sessions[session_id]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup and shutdown events"""
    # Startup
    global groq_client, wiki_wiki
    
    print("🔧 Initializing Ragnosis AI System...")
    
    # Initialize security
    initialize_secure_tokens()
    
    # Initialize components
    groq_client = initialize_groq_client()
    wiki_wiki = initialize_wikipedia()
    
    print("🎉 Ragnosis AI Assistant is ready!")
    print("💬 Enhanced with continuous conversations and Wikipedia integration!")
    print("🌐 Web interface: http://localhost:7860")
    print("🔧 API documentation: /api/docs")
    
    yield  # App runs here
    
    # Shutdown (if needed)
    print("🔧 Shutting down Ragnosis AI System...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Ragnosis AI - Advanced Medical Assistant",
    description="Your friendly AI companion for medical information and health guidance",
    version="4.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced API Routes
@app.get("/", response_class=HTMLResponse)
async def chat_interface():
    """Serve the main chat interface"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Ragnosis AI Frontend not found. Please check static files.</h1>", status_code=404)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    groq_status = "connected" if groq_client else "disconnected"
    wiki_status = "connected" if wiki_wiki else "disconnected"
    
    return HealthResponse(
        status="healthy",
        timestamp=str(datetime.now()),
        version="4.0.0",
        groq_status=groq_status,
        wikipedia_status=wiki_status
    )

@app.get("/api/status", response_model=SystemStatus)
async def system_status(_: str = Depends(verify_token)):
    """Detailed system status (protected)"""
    return SystemStatus(
        groq_connected=groq_client is not None,
        wikipedia_connected=wiki_wiki is not None,
        startup_time=str(system_startup_time),
        total_queries_processed=query_counter
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask Ragnosis AI a question with conversation context"""
    global query_counter
    start_time = time.time()
    query_counter += 1
    
    print(f"💬 Question received: {request.question}")
    print(f"📝 Session ID: {request.session_id}")
    
    # Check system readiness
    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="Ragnosis AI is taking a quick break! Please try again in a moment. 🫠"
        )
    
    try:
        # Get or create session
        session_id, session_data = get_or_create_session(request.session_id)
        
        # Search Wikipedia for medical information
        wikipedia_results = []
        if any(keyword in request.question.lower() for keyword in 
               ['disease', 'symptom', 'treatment', 'medical', 'health', 'diagnosis', 'medicine', 'pain', 'illness', 'condition']):
            wikipedia_results = search_wikipedia_medical(request.question)
        
        # Prepare conversation history for context
        conversation_history = session_data['conversation_history']
        
        # Generate conversational AI response with context
        answer = generate_conversational_response(
            request.question, 
            wikipedia_results, 
            conversation_history,
            session_id
        )
        
        # Update conversation history
        session_data['conversation_history'].extend([
            {"role": "user", "content": request.question},
            {"role": "assistant", "content": answer}
        ])
        
        # Limit conversation history to last 20 messages to prevent context overflow
        if len(session_data['conversation_history']) > 20:
            session_data['conversation_history'] = session_data['conversation_history'][-20:]
        
        session_data['message_count'] += 1
        
        # Extract follow-up questions
        follow_up_questions = extract_follow_up_questions(answer, request.question)
        
        # Get Wikipedia source links
        sources = get_wikipedia_source_links(wikipedia_results)
        
        response_time = round(time.time() - start_time, 2)
        
        print(f"✅ AI response generated in {response_time}s")
        print(f"💾 Session {session_id} now has {session_data['message_count']} messages")
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            response_time=response_time,
            model_used="llama-3.1-8b-instant",
            follow_up_questions=follow_up_questions,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"😅 Oops! Ragnosis AI encountered a small hiccup. Please try again!"
        )

@app.post("/quick-chat")
async def quick_chat(message: dict):
    """Simple chat endpoint for casual conversation"""
    if not groq_client:
        return {"response": "🤖 Hi there! I'm Ragnosis AI. I'm currently getting ready to chat. Please check back in a moment!"}
    
    user_message = message.get("message", "")
    session_id = message.get("session_id", "")
    
    # Get or create session
    session_id, session_data = get_or_create_session(session_id)
    
    # Simple greetings and casual responses
    greetings = {
        "hi": "👋 Hello there! I'm Ragnosis AI, your friendly medical companion. How can I help you today?",
        "hello": "👋 Hey! Great to meet you! I'm here to chat about health topics or just have a friendly conversation. What's on your mind?",
        "hey": "😊 Hey there! I'm Ragnosis AI, ready to help with medical questions or just chat. How are you doing?",
        "how are you": "🤖 I'm functioning perfectly, thanks for asking! Just here waiting to help you with any health questions or have a nice chat. How about you?",
        "what can you do": "🎯 I'm your AI health companion! I can:\n• Answer medical questions\n• Provide health information\n• Chat about wellness topics\n• Offer friendly advice\n• And just be a good listener!",
        "thanks": "💖 You're very welcome! I'm always here to help. Feel free to ask me anything else!",
        "thank you": "😊 My pleasure! Happy to assist. What else can I help with?"
    }
    
    # Check for simple greetings
    lower_msg = user_message.lower().strip()
    for greeting, response in greetings.items():
        if greeting in lower_msg:
            # Update session
            session_data['conversation_history'].extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response}
            ])
            session_data['message_count'] += 1
            
            return {
                "response": response,
                "session_id": session_id
            }
    
    # For other messages, use AI
    try:
        prompt = f"""You are Ragnosis AI - a friendly, cheerful AI assistant. The user said: "{user_message}"

Respond in a warm, conversational tone. Use emojis occasionally. Be helpful and engaging. If it's a medical question, provide useful information. If it's casual chat, be friendly and natural.

Your response:"""
        
        messages = [{"role": "user", "content": prompt}]
        response = generate_ai_response(messages, temperature=0.8)
        
        # Update session
        session_data['conversation_history'].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ])
        session_data['message_count'] += 1
        
        return {
            "response": response,
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "response": "😅 Oops! I'm having trouble responding right now. Please try again in a moment!",
            "session_id": session_id
        }

@app.post("/auth/test")
async def test_auth(_: str = Depends(verify_token)):
    """Test authentication (protected)"""
    return {"message": "Ragnosis AI authentication successful! 🎉", "status": "valid"}

# Error handlers
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "😅 Oops! Ragnosis AI is taking a quick coffee break. Please try again soon!"}
    )

@app.exception_handler(429)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "⏳ Whoa there! Let's slow down a bit. Please wait a moment before sending more messages."}
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "🔍 Hmm, I couldn't find what you're looking for. Let's try something else!"}
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info"
    )
