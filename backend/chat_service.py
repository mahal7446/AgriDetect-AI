"""
Chat service for AI-powered agricultural assistant using Google Gemini
Provides context-aware multilingual advice for plant disease management
Optimized for fast responses with caching, token limits, and generation config
"""
import os
from google import genai
from google.genai import types
from typing import Dict, Optional
import hashlib
import time

# Language names for prompt context
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'te': 'Telugu',
    'ta': 'Tamil',
    'bn': 'Bengali'
}

# Response cache (message-level + greeting-level)
_response_cache = {}
_greeting_cache = {}

# Generation config for fast, concise responses
_fast_gen_config = types.GenerateContentConfig(
    max_output_tokens=300,       # Enough for a complete answer without truncation
    temperature=0.4,             # Lower = faster, more deterministic
    top_p=0.8,                   # Narrow sampling for speed
    top_k=20,                    # Fewer candidates = faster
)

_greeting_gen_config = types.GenerateContentConfig(
    max_output_tokens=60,        # Greetings are very short
    temperature=0.3,
    top_p=0.8,
    top_k=10,
)


class ChatService:
    """Service for handling AI chat interactions with agricultural context"""
    
    def __init__(self):
        """Initialize Gemini AI client"""
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('VITE_GEMINI_API_KEY')
        if not api_key:
            print("[ERROR] GEMINI_API_KEY not found!")
            raise ValueError("No AI API key found in environment variables")
        print(f"[OK] GEMINI_API_KEY found: {api_key[:10]}...")
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'models/gemini-2.5-flash-lite'  # Fastest Gemini model with generous free tier
        
    def build_system_prompt(self, context: Dict, language: str) -> str:
        """Build concise agricultural expert prompt"""
        crop = context.get('crop', 'Unknown')
        disease = context.get('disease', 'None')
        lang = LANGUAGE_NAMES.get(language, 'English')
        
        has_context = disease not in ('None', 'Unknown') and crop != 'General'
        
        if has_context:
            return f"Agricultural expert. {crop} has {disease}. Give 2-3 sentence treatment/prevention advice in {lang}."
        return f"Agricultural expert. Answer farming questions in 2-3 sentences in {lang}."
    
    def get_initial_greeting(self, context: Dict, language: str) -> str:
        """Generate or return cached initial greeting"""
        crop = context.get('crop', 'plant')
        disease = context.get('disease', 'issue')
        lang = LANGUAGE_NAMES.get(language, 'English')
        
        # Check greeting cache
        cache_key = f"{crop}:{disease}:{language}"
        if cache_key in _greeting_cache:
            print(f"[OK] Cached greeting for {cache_key}")
            return _greeting_cache[cache_key]
        
        prompt = f"In {lang}, greet a farmer briefly (1 sentence). Their {crop} has {disease}. Invite questions."
        
        start = time.time()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=_greeting_gen_config
        )
        elapsed = time.time() - start
        
        text = getattr(response, 'text', '').strip()
        if not text:
            raise ValueError("AI greeting unavailable")
        
        # Cache greeting
        _greeting_cache[cache_key] = text
        print(f"[OK] Greeting ({elapsed:.1f}s): {text[:50]}...")
        return text
    
    def get_chat_response(
        self, 
        user_message: str, 
        context: Dict, 
        language: str = 'en',
        chat_history: Optional[list] = None
    ) -> Dict:
        """Get AI response with agricultural context — optimized for speed"""
        try:
            # Check cache first
            cache_key = hashlib.md5(
                f"{user_message}:{context}:{language}".encode()
            ).hexdigest()
            
            if cache_key in _response_cache:
                print(f"[OK] Cache hit: {user_message[:30]}...")
                return {
                    'success': True,
                    'response': _response_cache[cache_key],
                    'language': language,
                    'cached': True
                }
            
            system_prompt = self.build_system_prompt(context, language)
            full_prompt = f"{system_prompt}\n\nQ: {user_message}\nA:"
            
            start = time.time()
            print(f"[DEBUG] Sending to Gemini: {user_message[:40]}...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=_fast_gen_config
            )
            
            elapsed = time.time() - start
            response_text = getattr(response, 'text', '').strip()
            
            if not response_text:
                raise ValueError("Empty AI response")
            
            # Cache it
            _response_cache[cache_key] = response_text
            print(f"[OK] Response ({elapsed:.1f}s): {len(response_text)} chars")
            
            return {'success': True, 'response': response_text, 'language': language}
            
        except Exception as e:
            print(f"[ERROR] Chat failed: {str(e)}")
            return {'success': False, 'response': 'AI service unavailable', 'language': language}

_chat_service = None

def get_chat_service() -> ChatService:
    """Get or create ChatService singleton"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
