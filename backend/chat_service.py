"""
Chat service for AI-powered agricultural assistant using Google Gemini
Provides context-aware multilingual advice for plant disease management
"""
import os
import requests
from google import genai
from typing import Dict, Optional
import hashlib
import sys

# Language names for prompt context
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'te': 'Telugu',
    'ta': 'Tamil',
    'bn': 'Bengali'
}

# Simple response cache to handle quota limits
_response_cache = {}

class ChatService:
    """Service for handling AI chat interactions with agricultural context"""
    
    def __init__(self):
        """Initialize AI client. Prefer OpenAI if OPENAI_API_KEY is set, else Gemini"""
        # Force Gemini-only
        self.use_openai = False

        # (Gemini initialization handled above when OpenAI key is not present)

        if not self.use_openai:
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('VITE_GEMINI_API_KEY')
            if not api_key:
                print("[ERROR] GEMINI_API_KEY environment variable not found and OPENAI_API_KEY not set!")
                raise ValueError("No AI API key found in environment variables")
            print(f"[OK] GEMINI_API_KEY found: {api_key[:10]}...")
            self.client = genai.Client(api_key=api_key)
            self.model_name = 'models/gemini-flash-latest'
        
    def build_system_prompt(self, context: Dict, language: str) -> str:
        """Build agricultural expert system prompt"""
        crop = context.get('crop', 'Unknown')
        disease = context.get('disease', 'None')
        language_name = LANGUAGE_NAMES.get(language, 'English')
        
        has_disease_context = disease != 'None' and disease != 'Unknown' and crop != 'General'
        
        if has_disease_context:
            system_prompt = f"""You are an agricultural expert. Context: {crop} with {disease}.

Provide brief, practical advice (3-4 sentences max):
- Treatment steps
- Prevention tips
- Fertilizer if relevant

Respond ONLY in {language_name}. Be direct and farmer-friendly."""
        else:
            system_prompt = f"""You are an agricultural expert helping farmers.

Answer briefly (3-4 sentences max) about farming, crops, diseases, fertilizers.

Respond ONLY in {language_name}. Be practical and direct."""
        
        return system_prompt
    
    def get_initial_greeting(self, context: Dict, language: str) -> str:
        """Generate initial greeting message"""
        crop = context.get('crop', 'plant')
        disease = context.get('disease', 'issue')
        language_name = LANGUAGE_NAMES.get(language, 'English')
        prompt = (
            f"Respond in {language_name}. "
            f"Give a brief friendly greeting (1-2 sentences) as an agricultural assistant. "
            f"Acknowledge that the user's {crop} has {disease} and invite them to ask questions."
        )
        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        text = getattr(response, 'text', '').strip()
        if not text:
            raise ValueError("AI greeting unavailable")
        return text
    
    def get_chat_response(
        self, 
        user_message: str, 
        context: Dict, 
        language: str = 'en',
        chat_history: Optional[list] = None
    ) -> Dict:
        """Get AI response with agricultural context"""
        try:
            # Create cache key
            cache_key = hashlib.md5(
                f"{user_message}:{context}:{language}".encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in _response_cache:
                print(f"[OK] Cached response for: {user_message[:30]}...")
                return {
                    'success': True,
                    'response': _response_cache[cache_key],
                    'language': language,
                    'cached': True
                }
            
            system_prompt = self.build_system_prompt(context, language)
            full_prompt = f"{system_prompt}\n\nQ: {user_message}\n\nA:"
            
            print(f"[DEBUG] Sending to AI provider: {user_message[:40]}...")
            
            
            # If using OpenAI, call their API
            if getattr(self, 'use_openai', False):
                try:
                    headers = {
                        'Authorization': f'Bearer {self.openai_key}',
                        'Content-Type': 'application/json'
                    }
                    payload = {
                        'model': 'gpt-3.5-turbo',
                        'messages': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_message}
                        ],
                        'max_tokens': 200,
                        'temperature': 0.7,
                    }
                    resp = requests.post('https://api.openai.com/v1/chat/completions', json=payload, headers=headers, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        response_text = data['choices'][0]['message']['content'].strip()
                        # Cache and return
                        _response_cache[cache_key] = response_text
                        return {'success': True, 'response': response_text, 'language': language}
                    else:
                        api_error = Exception(f'OpenAI API error: {resp.status_code} {resp.text}')
                        raise api_error
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    error_type = type(api_error).__name__
                    print(f"[DEBUG] API Error: {error_type}")
                    print(f"[DEBUG] Message: {error_str[:80]}")
                    # Quota exhausted - use fallback
                    if "quota" in error_str or "429" in error_str or "rate limit" in error_str:
                        print(f"[WARNING] OpenAI quota/rate limit")
                        pass
                    # otherwise fallthrough to Gemini path if available
            # Not using OpenAI or OpenAI failed - use Gemini if configured
            try:
                response = self.client.models.generate_content(model=self.model_name, contents=full_prompt)
            except Exception as api_error:
                error_str = str(api_error).lower()
                error_type = type(api_error).__name__
                print(f"[DEBUG] API Error: {error_type}")
                print(f"[DEBUG] Message: {error_str[:80]}")
                # Quota exhausted or other errors
                if "quota" in error_str or "429" in error_str or "resource_exhausted" in error_str or "ResourceExhausted" in error_type:
                    print(f"[WARNING] Quota exceeded or AI error")
                    raise api_error
                # Retry with simpler prompt
                print("[INFO] Retrying with simpler prompt...")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"Answer in {LANGUAGE_NAMES.get(language, 'English')}: {user_message}"
                )
            
            response_text = getattr(response, 'text', '').strip()
            
            if not response_text:
                raise ValueError("Empty AI response")
            
            # Cache it
            _response_cache[cache_key] = response_text
            print(f"[OK] Response: {len(response_text)} chars")
            
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
