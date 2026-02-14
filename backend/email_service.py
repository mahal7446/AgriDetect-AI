import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.smtp_email = os.getenv('SMTP_EMAIL')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.app_name = "AgriDetect AI"
        
    def send_verification_email(self, to_email, verification_token, user_name):
        """Send email verification link to user"""
        
        # Create verification URL
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:8081')
        verification_url = f"{frontend_url}/verify-email/{verification_token}"
        
        # Create HTML email template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .container {{
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    border-radius: 16px;
                    padding: 40px;
                    color: white;
                }}
                .logo {{
                    font-size: 28px;
                    font-weight: bold;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .content {{
                    background: white;
                    color: #333;
                    padding: 30px;
                    border-radius: 12px;
                    margin-top: 20px;
                }}
                .button {{
                    display: inline-block;
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    padding: 14px 32px;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    margin: 20px 0;
                    text-align: center;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: rgba(255,255,255,0.8);
                    font-size: 14px;
                }}
                .warning {{
                    background: #fef3c7;
                    border-left: 4px solid #f59e0b;
                    padding: 12px;
                    margin: 20px 0;
                    border-radius: 4px;
                    color: #92400e;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="logo">🌾 {self.app_name}</div>
                
                <div class="content">
                    <h2 style="color: #10b981; margin-top: 0;">Welcome, {user_name}! 🌱</h2>
                    
                    <p>Thank you for joining <strong>{self.app_name}</strong>! We're excited to help you protect your crops with AI-powered disease detection.</p>
                    
                    <p>To complete your registration and activate your account, please verify your email address by clicking the button below:</p>
                    
                    <div style="text-align: center;">
                        <a href="{verification_url}" class="button">
                            Verify Email Address
                        </a>
                    </div>
                    
                    <p style="color: #666; font-size: 14px;">Or copy and paste this link into your browser:</p>
                    <p style="background: #f3f4f6; padding: 12px; border-radius: 6px; word-break: break-all; font-size: 13px;">
                        {verification_url}
                    </p>
                    
                    <div class="warning">
                        ⏱️ <strong>Note:</strong> This verification link will expire in <strong>24 hours</strong>.
                    </div>
                    
                    <p>If you didn't create an account with {self.app_name}, please ignore this email.</p>
                </div>
                
                <div class="footer">
                    <p>© 2024 {self.app_name} | Helping farmers protect their crops</p>
                    <p style="font-size: 12px;">This is an automated email. Please do not reply.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create message
        message = MIMEMultipart('alternative')
        message['Subject'] = f"Verify your {self.app_name} email address 🌾"
        message['From'] = f"{self.app_name} <{self.smtp_email}>"
        message['To'] = to_email
        
        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        message.attach(html_part)
        
        try:
            # Send email via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_email, self.smtp_password)
                server.send_message(message)
                
            return True, "Verification email sent successfully"
            
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False, f"Failed to send email: {str(e)}"
    
    def send_welcome_email(self, to_email, user_name):
        """Send welcome email after successful verification"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .container {{
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    border-radius: 16px;
                    padding: 40px;
                    color: white;
                }}
                .content {{
                    background: white;
                    color: #333;
                    padding: 30px;
                    border-radius: 12px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div style="font-size: 28px; font-weight: bold; text-align: center; margin-bottom: 20px;">
                    🌾 {self.app_name}
                </div>
                
                <div class="content">
                    <h2 style="color: #10b981;">Welcome to {self.app_name}! 🎉</h2>
                    <p>Hi {user_name},</p>
                    <p>Your email has been verified successfully! You can now access all features of {self.app_name}.</p>
                    <p><strong>Get started with:</strong></p>
                    <ul>
                        <li>📸 Upload plant images for disease detection</li>
                        <li>🌤️ Check real-time weather conditions</li>
                        <li>📊 View your scan history and analytics</li>
                        <li>👥 Join our farming community</li>
                    </ul>
                    <p>Happy farming! 🌱</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        message = MIMEMultipart('alternative')
        message['Subject'] = f"Welcome to {self.app_name}! 🎉"
        message['From'] = f"{self.app_name} <{self.smtp_email}>"
        message['To'] = to_email
        
        html_part = MIMEText(html_content, 'html')
        message.attach(html_part)
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_email, self.smtp_password)
                server.send_message(message)
            return True
        except Exception as e:
            print(f"Error sending welcome email: {str(e)}")
            return False

# Create singleton instance
email_service = EmailService()
