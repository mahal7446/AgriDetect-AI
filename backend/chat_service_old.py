"""
Chat service for AI-powered agricultural assistant using Google Gemini
Provides context-aware multilingual advice for plant disease management
"""
import os
import google.generativeai as genai
from typing import Dict, Optional

# Language names for prompt context
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'te': 'Telugu',
    'ta': 'Tamil',
    'bn': 'Bengali'
}

class ChatService:
    """Service for handling AI chat interactions with agricultural context"""
    
    def __init__(self):
        """Initialize Gemini API client"""
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('VITE_GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY or VITE_GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Configure model with optimized settings for faster responses
        generation_config = {
            'temperature': 0.7,  # Balanced creativity
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 200,  # Limit response length for speed
            'candidate_count': 1,
        }
        
        self.model = genai.GenerativeModel(
            'gemini-flash-latest',  # Use gemini-flash-latest for reliability and speed
            generation_config=generation_config
        )
        
    def build_system_prompt(self, context: Dict, language: str) -> str:
        """Build concise agricultural expert system prompt"""
        crop = context.get('crop', 'Unknown')
        disease = context.get('disease', 'None')
        language_name = LANGUAGE_NAMES.get(language, 'English')
        
        # Check if this is a specific disease context or general query
        has_disease_context = disease != 'None' and disease != 'Unknown' and crop != 'General'
        
        if has_disease_context:
            # Specific disease context - concise prompt
            system_prompt = f"""You are an agricultural expert. Context: {crop} with {disease}.

Provide brief, practical advice (3-4 sentences max):
- Treatment steps
- Prevention tips
- Fertilizer if relevant

Respond ONLY in {language_name}. Be direct and farmer-friendly."""
        
        else:
            # General agriculture - concise prompt
            system_prompt = f"""You are an agricultural expert helping farmers.

Answer briefly (3-4 sentences max) about:
farming, crops, diseases, fertilizers, irrigation, soil, pests.

Respond ONLY in {language_name}. Be practical and direct."""
        
        return system_prompt
    
    def get_initial_greeting(self, context: Dict, language: str) -> str:
        """
        Generate initial greeting message based on disease detection
        
        Args:
            context: Dictionary containing crop and disease info
            language: Target language code
        
        Returns:
            Greeting message string
        """
        crop = context.get('crop', 'plant')
        disease = context.get('disease', 'issue')
        
        greetings = {
            'en': f"Hello! I can see your {crop} has {disease}. I'm here to help you treat and prevent this disease. What would you like to know?",
            'hi': f"नमस्ते! मैं देख सकता हूं कि आपके {crop} में {disease} है। मैं इस बीमारी के इलाज और रोकथाम में आपकी मदद करने के लिए यहां हूं। आप क्या जानना चाहेंगे?",
            'kn': f"ನಮಸ್ಕಾರ! ನಿಮ್ಮ {crop} ಗೆ {disease} ಇದೆ ಎಂದು ನಾನು ನೋಡಬಹುದು. ಈ ರೋಗವನ್ನು ಚಿಕಿತ್ಸೆ ಮಾಡಲು ಮತ್ತು ತಡೆಗಟ್ಟಲು ನಾನು ಇಲ್ಲಿದ್ದೇನೆ. ನೀವು ಏನು ತಿಳಿದುಕೊಳ್ಳಲು ಬಯಸುತ್ತೀರಿ?",
            'te': f"నమస్కారం! మీ {crop} కి {disease} ఉందని నేను చూడగలను. ఈ వ్యాధికి చికిత్స చేయడానికి మరియు నిరోధించడానికి నేను ఇక్కడ ఉన్నాను. మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
            'ta': f"வணக்கம்! உங்கள் {crop} க்கு {disease} உள்ளது என்பதை நான் காண்கிறேன். இந்த நோயைக் குணப்படுத்தவும் தடுக்கவும் நான் இங்கே இருக்கிறேன். நீங்கள் என்ன தெரிந்து கொள்ள விரும்புகிறீர்கள்?",
            'bn': f"নমস্কার! আমি দেখতে পাচ্ছি আপনার {crop} এ {disease} আছে। এই রোগের চিকিৎসা এবং প্রতিরোধে সাহায্য করতে আমি এখানে আছি। আপনি কী জানতে চান?"
        }
        
        return greetings.get(language, greetings['en'])
    
    def get_chat_response(
        self, 
        user_message: str, 
        context: Dict, 
        language: str = 'en',
        chat_history: Optional[list] = None
    ) -> Dict:
        """
        Get AI response for user message with agricultural context
        
        Args:
            user_message: User's question or message
            context: Disease detection context
            language: Target language code
            chat_history: Previous conversation history (optional)
        
        Returns:
            Dictionary with response text and metadata
        """
        try:
            # Build system prompt with context
            system_prompt = self.build_system_prompt(context, language)
            
            # Combine system prompt and user message concisely
            full_prompt = f"{system_prompt}\n\nQ: {user_message}\n\nA:"
            
            print(f"[DEBUG] [ChatService] Sending request to Gemini...")
            print(f"[DEBUG] [ChatService] Model: {self.model.model_name}")
            print(f"[DEBUG] [ChatService] Prompt: {user_message[:50]}...")
            
            # Configure safety settings to be more permissive for agricultural advice
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            # Generate response with timeout consideration
            response = self.model.generate_content(
                full_prompt,
                safety_settings=safety_settings
            )
            
            # Check if response was blocked
            if not response.candidates or response.candidates[0].finish_reason != 1:
                reason = "Unknown"
                if response.candidates:
                    reason = response.candidates[0].finish_reason
                print(f"[DEBUG] [ChatService] Response blocked or incomplete. Finish Reason: {reason}")
                
                # Check for safety ratings if blocked
                if hasattr(response, 'prompt_feedback'):
                    print(f"[DEBUG] [ChatService] Prompt Feedback: {response.prompt_feedback}")
                
                raise ValueError(f"Response blocked by safety filters (Reason: {reason})")

            # Extract text from response
            response_text = response.text.strip()
            
            print(f"[DEBUG] [ChatService] Received response from Gemini. Length: {len(response_text)}")
            
            return {
                'success': True,
                'response': response_text,
                'language': language
            }
            
        except Exception as e:
            print(f"[ERROR] Chat service failed: {str(e)}")
            
            # Fallback error messages in different languages
            error_messages = {
                'en': "I'm having trouble connecting right now. Please try again in a moment.",
                'hi': "मुझे अभी कनेक्ट करने में परेशानी हो रही है। कृपया एक पल में पुनः प्रयास करें।",
                'kn': "ನನಗೆ ಈಗ ಸಂಪರ್ಕಿಸಲು ತೊಂದರೆಯಾಗುತ್ತಿದೆ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
                'te': "నాకు ఇప్పుడు కనెక్ట్ చేయడంలో సమస్య ఉంది. దయచేసి ఒక క్షణంలో మళ్లీ ప్రయత్నించండి.",
                'ta': "இப்போது எனக்கு இணைப்பதில் சிக்கல் உள்ளது. தயவுசெய்து சிறிது நேரத்தில் மீண்டும் முயற்சிக்கவும்.",
                'bn': "আমার এখন সংযোগ করতে সমস্যা হচ্ছে। অনুগ্রহ করে একটু পরে আবার চেষ্টা করুন।"
            }
            
            return {
                'success': False,
                'response': error_messages.get(language, error_messages['en']),
                'error': str(e)
            }

# Singleton instance
_chat_service = None

def get_chat_service() -> ChatService:
    """Get or create ChatService singleton instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
