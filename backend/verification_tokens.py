from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class TokenManager:
    def __init__(self):
        self.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
        self.serializer = URLSafeTimedSerializer(self.secret_key)
        self.token_expiration = 86400  # 24 hours in seconds
        
        # In-memory store for used tokens (in production, use database)
        self.used_tokens = set()
        
    def generate_token(self, email):
        """Generate a verification token for an email"""
        return self.serializer.dumps(email, salt='email-verification')
    
    def verify_token(self, token, max_age=None):
        """
        Verify a token and return the email if valid
        Returns: (success, email_or_error_message)
        """
        if max_age is None:
            max_age = self.token_expiration
            
        # Check if token already used
        if token in self.used_tokens:
            return False, "This verification link has already been used"
        
        try:
            email = self.serializer.loads(
                token,
                salt='email-verification',
                max_age=max_age
            )
            
            # Mark token as used
            self.used_tokens.add(token)
            
            return True, email
            
        except SignatureExpired:
            return False, "Verification link has expired. Please request a new one."
        except BadSignature:
            return False, "Invalid verification link"
        except Exception as e:
            return False, f"Error verifying token: {str(e)}"
    
    def invalidate_token(self, token):
        """Mark a token as used/invalid"""
        self.used_tokens.add(token)

# Create singleton instance
token_manager = TokenManager()
