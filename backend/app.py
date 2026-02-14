"""
Flask backend server for plant disease detection using ML model (.h5)
"""
# Suppress TensorFlow warnings and verbose output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv
from pathlib import Path
from werkzeug.utils import secure_filename
from database import (
    init_db, create_user, get_user, verify_password,
    save_scan, get_user_scans, delete_scan, get_scan_by_id,
    update_user_profile, update_profile_picture,
    create_alert, get_recent_alerts, get_alerts_by_location,
    delete_alert, update_alert, get_user_notification_preference,
    update_user_notification_preference, get_new_alerts_count,
    get_user_stats, get_user_accuracy, get_total_users_count,
    get_analytics_summary, get_analytics_charts, get_analytics_reports
)
#         from working_model_manager import get_working_model_manager as get_model_manager as get_model_manager
from verification_tokens import token_manager
from email_service import EmailService
from chat_service import get_chat_service
from working_model_manager import get_working_model_manager

load_dotenv()
_root_env = (Path(__file__).resolve().parent.parent / '.env')
if _root_env.exists():
    load_dotenv(dotenv_path=_root_env, override=False)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Working model manager (simplified, no issues)
working_model_manager = None

# In-memory users database (should be replaced with proper database)
users_db = {}

# Upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads', 'profile_pictures')
ALERT_IMAGES_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads', 'alert_images')
SCAN_IMAGES_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads', 'scan_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ALERT_IMAGES_FOLDER, exist_ok=True)
os.makedirs(SCAN_IMAGES_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_models():
    global working_model_manager
    working_model_manager = get_working_model_manager()
    return True

def preprocess_image(image_file):
    """
    Preprocess image for model prediction
    Your EfficientNet-B3 model has built-in preprocessing layers (Rescaling, Normalization),
    so we pass raw pixel values (0-255) instead of normalized values.
    """
    # Read image from file
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match model input size (224x224)
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Convert to numpy array - keep raw pixel values (0-255) since model has built-in preprocessing
    img_array = np.array(image, dtype=np.float32)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_class_labels():
    """
    Return class labels for predictions
    Your model has 9 classes - UPDATE THESE with your actual class names
    IMPORTANT: The order must match your model's training labels exactly!
    """
    # TODO: Replace these with your actual 9 class names in the correct order
    return [
        "Blackgram_LeafCrinckle",
        "Blackgram_YellowMosaic",
        "Blackgram_Healthy",
        "Blackgram_Anthracnose",
        "Blackgram_PowderyMildew",
        "Corn_CommonRust",
        "Corn_Blight",
        "Corn_Healthy",
        "Corn_GrayLeafSpot",
    ]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": working_model_manager is not None and len(working_model_manager.models) > 0,
        "num_models": len(working_model_manager.models) if working_model_manager else 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        image_file = request.files['image']
        if working_model_manager is None:
            init_models()
        image = Image.open(io.BytesIO(image_file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        result = working_model_manager.predict_two_stage(image)
        if not result.get('success'):
            return jsonify({"success": False, "error": result.get('error', 'Prediction failed')}), 500
        disease_name = result.get('disease', 'Unknown')
        confidence = float(result.get('confidence', 0.0)) * 100.0
        crop_name = result.get('cropName', 'Unknown')
        severity = result.get('severity', 'Low')
        top_candidates = result.get('topCandidates', [])
        return jsonify({
            "success": True,
            "diseaseName": disease_name,
            "confidence": round(confidence, 2),
            "cropName": crop_name,
            "severity": severity,
            "topCandidates": [
                {"disease": c["disease"], "confidence": round(c["confidence"] * 100.0, 2)}
                for c in top_candidates
            ],
            "description": f"{crop_name} - {disease_name} detected with {confidence:.2f}% confidence."
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

# ============== VALIDATION FUNCTIONS ==============

def validate_password(password):
    """
    Validate password meets security requirements:
    - 8-12 characters length
    - Contains uppercase letters (A-Z)
    - Contains lowercase letters (a-z)
    - Contains numbers (0-9)
    - Contains special characters (!, @, #, $, %, etc.)
    """
    import re
    
    if not password:
        return False, "Password is required"
    
    # Check length
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if len(password) > 12:
        return False, "Password must not exceed 12 characters"
    
    # Check for uppercase letter
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter (A-Z)"
    
    # Check for lowercase letter
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter (a-z)"
    
    # Check for digit
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number (0-9)"
    
    # Check for special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;/`~]', password):
        return False, "Password must contain at least one special character (!, @, #, $, %, etc.)"
    
    return True, "Password is valid"

def validate_phone(phone):
    """
    Validate phone number:
    - Exactly 10 digits
    - Only numeric characters (spaces and dashes are removed)
    """
    import re
    
    if not phone:
        return False, "Phone number is required"
    
    # Remove spaces, dashes, and other common separators
    cleaned_phone = re.sub(r'[\s\-().]', '', phone)
    
    # Check if it contains only digits
    if not cleaned_phone.isdigit():
        return False, "Phone number must contain only digits"
    
    # Check length
    if len(cleaned_phone) != 10:
        return False, "Phone number must be exactly 10 digits"
    
    return True, cleaned_phone

# ============== AUTHENTICATION ENDPOINTS ==============

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Handle user signup - create account in database"""
    try:
        data = request.json
        email = data.get('email')
        full_name = data.get('fullName')
        phone = data.get('phone', '')
        password = data.get('password')
        
        # Validate input
        if not all([email, full_name, password, phone]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Validate password
        password_valid, password_message = validate_password(password)
        if not password_valid:
            return jsonify({"error": password_message}), 400
        
        # Validate phone number
        phone_valid, phone_result = validate_phone(phone)
        if not phone_valid:
            return jsonify({"error": phone_result}), 400
        
        # Use cleaned phone number
        cleaned_phone = phone_result
        
        # Create user in database (password will be hashed automatically)
        success, result = create_user(email, full_name, cleaned_phone, password)
        
        if not success:
            return jsonify({"error": result}), 409 if "already registered" in result else 500
        
        return jsonify({
            "success": True,
            "message": "Account created successfully! You can now log in.",
            "user": result
        }), 200
        
    except Exception as e:
        print(f"Signup error: {str(e)}")
        return jsonify({"error": f"Signup failed: {str(e)}"}), 500

@app.route('/api/auth/verify-email/<token>', methods=['GET'])
def verify_email(token):
    """Verify email using token"""
    try:
        # Verify token
        success, result = token_manager.verify_token(token)
        
        if not success:
            return jsonify({
                "success": False,
                "error": result
            }), 400
        
        email = result
        
        # Check if user data exists in pending
        if email not in pending_verifications:
            return jsonify({
                "success": False,
                "error": "Verification data not found"
            }), 404
        
        # Move user from pending to active users
        user_data = pending_verifications.pop(email)
        users_db[email] = {
            'email': user_data['email'],
            'fullName': user_data['fullName'],
            'phone': user_data['phone'],
            'password': user_data['password'],
            'verified': True,
            'created_at': str(np.datetime64('now'))
        }
        
        # Send welcome email
        email_service.send_welcome_email(email, user_data['fullName'])
        
        return jsonify({
            "success": True,
            "message": "Email verified successfully! You can now log in.",
            "user": {
                "email": email,
                "fullName": user_data['fullName']
            }
        }), 200
        
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Verification failed: {str(e)}"
        }), 500

@app.route('/api/auth/resend-verification', methods=['POST'])
def resend_verification():
    """Resend verification email"""
    try:
        data = request.json
        email = data.get('email')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        # Check if user is in pending verifications
        if email not in pending_verifications:
            return jsonify({"error": "No pending verification for this email"}), 404
        
        # Generate new token
        token = token_manager.generate_token(email)
        pending_verifications[email]['token'] = token
        
        # Resend email
        success, message = email_service.send_verification_email(
            to_email=email,
            verification_token=token,
            user_name=pending_verifications[email]['fullName']
        )
        
        if not success:
            return jsonify({"error": message}), 500
        
        return jsonify({
            "success": True,
            "message": "Verification email resent successfully!"
        }), 200
        
    except Exception as e:
        print(f"Resend error: {str(e)}")
        return jsonify({"error": f"Resend failed: {str(e)}"}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Handle user login with database"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({"error": "Missing email or password"}), 400
        
        # Get user from database
        user = get_user(email)
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Verify password using bcrypt
        if not verify_password(user['password_hash'], password):
            return jsonify({"error": "Invalid credentials"}), 401
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "email": user['email'],
                "fullName": user['fullName'],
                "phone": user['phone'],
                "address": user.get('address'),
                "profilePictureUrl": user.get('profilePictureUrl')
            }
        }), 200
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({"error": f"Login failed: {str(e)}"}), 500

# ============== PROFILE ENDPOINTS ==============

@app.route('/api/profile/get', methods=['GET'])
def get_profile():
    """Get user profile by email"""
    try:
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        user = get_user(email)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Don't send password hash to frontend
        user.pop('password_hash', None)
        
        return jsonify({
            "success": True,
            "user": user
        }), 200
        
    except Exception as e:
        print(f"Get profile error: {str(e)}")
        return jsonify({"error": f"Failed to fetch profile: {str(e)}"}), 500

@app.route('/api/profile/update', methods=['POST'])
def update_profile():
    """Update user profile information"""
    try:
        data = request.json
        email = data.get('email')
        full_name = data.get('fullName')
        phone = data.get('phone')
        address = data.get('address')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        # Validate phone if provided
        if phone:
            phone_valid, phone_result = validate_phone(phone)
            if not phone_valid:
                return jsonify({"error": phone_result}), 400
            phone = phone_result  # Use cleaned phone
        
        # Update profile
        success, message = update_user_profile(email, full_name, phone, address)
        
        if not success:
            return jsonify({"error": message}), 500
        
        # Get updated user data
        user = get_user(email)
        if user:
            user.pop('password_hash', None)
        
        return jsonify({
            "success": True,
            "message": message,
            "user": user
        }), 200
        
    except Exception as e:
        print(f"Update profile error: {str(e)}")
        return jsonify({"error": f"Failed to update profile: {str(e)}"}), 500

@app.route('/api/profile/upload-picture', methods=['POST'])
def upload_profile_picture():
    """Upload user profile picture"""
    try:
        email = request.form.get('email')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, gif"}), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({"error": "File too large. Maximum size is 5MB"}), 400
        
        # Generate secure filename
        filename = secure_filename(file.filename)
        # Add email prefix to avoid conflicts
        safe_email = email.replace('@', '_').replace('.', '_')
        unique_filename = f"{safe_email}_{filename}"
        
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Generate URL path for the uploaded file
        profile_picture_url = f"/uploads/profile_pictures/{unique_filename}"
        
        # Update database
        success, message = update_profile_picture(email, profile_picture_url)
        
        if not success:
            # Remove uploaded file if database update fails
            os.remove(filepath)
            return jsonify({"error": message}), 500
        
        return jsonify({
            "success": True,
            "message": "Profile picture uploaded successfully",
            "profilePictureUrl": profile_picture_url
        }), 200
        
    except Exception as e:
        print(f"Upload picture error: {str(e)}")
        return jsonify({"error": f"Failed to upload picture: {str(e)}"}), 500

@app.route('/uploads/profile_pictures/<filename>')
def serve_profile_picture(filename):
    """Serve uploaded profile pictures"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/profile/stats', methods=['GET'])
def get_stats():
    """Get scan statistics for a user"""
    try:
        email = request.args.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        stats = get_user_stats(email)
        return jsonify({
            "success": True,
            "stats": stats
        }), 200
    except Exception as e:
        print(f"Get stats error: {str(e)}")
        return jsonify({"error": f"Failed to fetch stats: {str(e)}"}), 500

@app.route('/api/profile/accuracy', methods=['GET'])
def get_accuracy():
    """Get user's average accuracy (average confidence across all scans)"""
    try:
        email = request.args.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400

        accuracy = get_user_accuracy(email)
        return jsonify({
            "success": True,
            "accuracy": round(accuracy, 2)
        }), 200
    except Exception as e:
        print(f"Get accuracy error: {str(e)}")
        return jsonify({"error": f"Failed to fetch accuracy: {str(e)}"}), 500

@app.route('/api/stats/total-users', methods=['GET'])
def get_total_users_endpoint():
    """Get total number of registered users"""
    return jsonify(get_total_users_count())

# ============== COMMUNITY ALERTS ENDPOINTS ==============

@app.route('/api/alerts/recent', methods=['GET'])
def get_alerts():
    """Get recent community alerts"""
    try:
        limit = request.args.get('limit', 10, type=int)
        alerts = get_recent_alerts(limit)
        
        return jsonify({
            "success": True,
            "alerts": alerts
        }), 200
        
    except Exception as e:
        print(f"Get alerts error: {str(e)}")
        return jsonify({"error": f"Failed to fetch alerts: {str(e)}"}), 500

@app.route('/api/alerts/submit', methods=['POST'])
def submit_alert():
    """Submit a new community alert with optional photo"""
    try:
        farmer_name = request.form.get('farmerName')
        location = request.form.get('location')
        disease_reported = request.form.get('diseaseReported')
        description = request.form.get('description', '')
        prevention_methods = request.form.get('preventionMethods', '')
        user_email = request.form.get('userEmail')
        
        # Validate required fields
        if not all([farmer_name, location, disease_reported]):
            return jsonify({"error": "Missing required fields: farmerName, location, diseaseReported"}), 400
        
        image_url = None
        
        # Handle optional image upload
        if 'image' in request.files:
            file = request.files['image']
            
            if file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, gif"}), 400
                
                # Check file size
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > MAX_FILE_SIZE:
                    return jsonify({"error": "File too large. Maximum size is 5MB"}), 400
                
                # Generate secure filename
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                import time
                unique_filename = f"{int(time.time())}_{filename}"
                
                filepath = os.path.join(ALERT_IMAGES_FOLDER, unique_filename)
                file.save(filepath)
                
                # Generate URL path for the uploaded file
                image_url = f"/uploads/alert_images/{unique_filename}"
        
        # Create alert in database
        success, result = create_alert(
            farmer_name=farmer_name,
            location=location,
            disease_reported=disease_reported,
            description=description,
            prevention_methods=prevention_methods,
            image_url=image_url,
            user_email=user_email
        )
        
        if not success:
            # Remove uploaded file if database insert fails
            if image_url:
                filepath = os.path.join(ALERT_IMAGES_FOLDER, unique_filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            return jsonify({"error": result}), 500
        
        return jsonify({
            "success": True,
            "message": "Alert submitted successfully",
            "alert": result
        }), 200
        
    except Exception as e:
        print(f"Submit alert error: {str(e)}")
        return jsonify({"error": f"Failed to submit alert: {str(e)}"}), 500

@app.route('/uploads/alert_images/<filename>')
def serve_alert_image(filename):
    """Serve uploaded alert images"""
    return send_from_directory(ALERT_IMAGES_FOLDER, filename)

@app.route('/api/alerts/by-location', methods=['GET'])
def get_alerts_location():
    """Get alerts filtered by user's location"""
    try:
        email = request.args.get('email')
        limit = request.args.get('limit', 20, type=int)
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        alerts = get_alerts_by_location(email, limit)
        
        return jsonify({
            "success": True,
            "alerts": alerts
        }), 200
        
    except Exception as e:
        print(f"Get alerts by location error: {str(e)}")
        return jsonify({"error": f"Failed to fetch alerts: {str(e)}"}), 500

@app.route('/api/alerts/delete/<int:alert_id>', methods=['DELETE'])
def delete_community_alert(alert_id):
    """Delete a community alert"""
    try:
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        success, result = delete_alert(alert_id, email)
        
        if not success:
            return jsonify({"error": result}), 404
            
        # Cleanup image file if exists
        if result and result.startswith('/uploads/alert_images/'):
            filename = result.replace('/uploads/alert_images/', '')
            filepath = os.path.join(ALERT_IMAGES_FOLDER, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Failed to delete alert image: {e}")
                    
        return jsonify({
            "success": True,
            "message": "Alert deleted successfully"
        }), 200
        
    except Exception as e:
        print(f"Delete alert error: {str(e)}")
        return jsonify({"error": f"Failed to delete alert: {str(e)}"}), 500

@app.route('/api/alerts/update/<int:alert_id>', methods=['POST'])
def update_community_alert(alert_id):
    """Update a community alert"""
    try:
        farmer_name = request.form.get('farmerName')
        location = request.form.get('location')
        disease_reported = request.form.get('diseaseReported')
        description = request.form.get('description')
        prevention_methods = request.form.get('preventionMethods')
        user_email = request.form.get('userEmail')
        
        if not user_email:
            return jsonify({"error": "User email is required"}), 400
            
        image_url = None
        # Handle optional new image upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                if allowed_file(file.filename):
                    import time
                    filename = secure_filename(file.filename)
                    unique_filename = f"{int(time.time())}_{filename}"
                    filepath = os.path.join(ALERT_IMAGES_FOLDER, unique_filename)
                    file.save(filepath)
                    image_url = f"/uploads/alert_images/{unique_filename}"

        success, message = update_alert(
            alert_id=alert_id,
            user_email=user_email,
            farmer_name=farmer_name,
            location=location,
            disease_reported=disease_reported,
            description=description,
            prevention_methods=prevention_methods,
            image_url=image_url
        )
        
        if not success:
            return jsonify({"error": message}), 500
            
        return jsonify({
            "success": True,
            "message": message
        }), 200
        
    except Exception as e:
        print(f"Update alert error: {str(e)}")
        return jsonify({"error": f"Failed to update alert: {str(e)}"}), 500

@app.route('/api/alerts/new-count', methods=['GET'])
def get_alerts_count():
    """Get count of new alerts since last seen"""
    try:
        email = request.args.get('email')
        last_seen_id = request.args.get('lastSeenId', 0, type=int)
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        count = get_new_alerts_count(email, last_seen_id)
        
        return jsonify({
            "success": True,
            "count": count
        }), 200
        
    except Exception as e:
        print(f"Get new alerts count error: {str(e)}")
        return jsonify({"error": f"Failed to fetch count: {str(e)}"}), 500

@app.route('/api/profile/notification-preference', methods=['GET'])
def get_notification_pref():
    """Get user notification preference"""
    try:
        email = request.args.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        success, enabled = get_user_notification_preference(email)
        return jsonify({
            "success": success,
            "enabled": enabled
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/update-notification-preference', methods=['POST'])
def update_notification_pref():
    """Update user notification preference"""
    try:
        data = request.json
        email = data.get('email')
        enabled = data.get('enabled')
        
        if not email or enabled is None:
            return jsonify({"error": "Email and enabled status are required"}), 400
            
        success, message = update_user_notification_preference(email, enabled)
        return jsonify({
            "success": success,
            "message": message
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============== HISTORY API ENDPOINTS ==============

@app.route('/api/history/save', methods=['POST'])
def save_history():
    """Save a scan to user's history with image"""
    try:
        user_email = request.form.get('email')
        disease_name = request.form.get('diseaseName')
        confidence = request.form.get('confidence')
        crop_name = request.form.get('cropName')
        severity = request.form.get('severity')
        
        # Validate required fields
        if not all([user_email, disease_name, confidence]):
            return jsonify({"error": "Missing required fields"}), 400
        
        image_url = None
        
        # Handle image upload
        if 'image' in request.files:
            file = request.files['image']
            
            if file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({"error": "Invalid file type"}), 400
                
                # Generate unique filename
                import time
                filename = secure_filename(file.filename)
                safe_email = user_email.replace('@', '_').replace('.', '_')
                unique_filename = f"{safe_email}_{int(time.time())}_{filename}"
                
                filepath = os.path.join(SCAN_IMAGES_FOLDER, unique_filename)
                file.save(filepath)
                
                image_url = f"/uploads/scan_images/{unique_filename}"
        
        # Save to database
        success, result = save_scan(
            user_email=user_email,
            disease_name=disease_name,
            confidence=float(confidence),
            crop_name=crop_name,
            severity=severity,
            image_url=image_url,
            risk_level=request.form.get('riskLevel'),
            health_status=request.form.get('healthStatus')
        )
        
        if not success:
            # Cleanup uploaded file if db save fails
            if image_url:
                filepath = os.path.join(SCAN_IMAGES_FOLDER, unique_filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            return jsonify({"error": result}), 500
        
        return jsonify({
            "success": True,
            "message": "Scan saved to history",
            "scanId": result,
            "imageUrl": image_url
        }), 200
        
    except Exception as e:
        print(f"Save history error: {str(e)}")
        return jsonify({"error": f"Failed to save history: {str(e)}"}), 500

@app.route('/api/history/get', methods=['GET'])
def get_history():
    """Get user's scan history"""
    try:
        email = request.args.get('email')
        limit = request.args.get('limit', 50, type=int)
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        scans = get_user_scans(email, limit)
        
        return jsonify({
            "success": True,
            "history": scans,
            "count": len(scans)
        }), 200
        
    except Exception as e:
        print(f"Get history error: {str(e)}")
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500

@app.route('/api/history/delete/<int:scan_id>', methods=['DELETE'])
def delete_history(scan_id):
    """Delete a scan from user's history"""
    try:
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        success, result = delete_scan(scan_id, email)
        
        if not success:
            return jsonify({"error": result}), 404
        
        # Cleanup image file if exists
        if result and result.startswith('/uploads/scan_images/'):
            filename = result.replace('/uploads/scan_images/', '')
            filepath = os.path.join(SCAN_IMAGES_FOLDER, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Deleted scan image: {filename}")
                except Exception as e:
                    print(f"Failed to delete image: {e}")
        
        return jsonify({
            "success": True,
            "message": "Scan deleted successfully"
        }), 200
        
    except Exception as e:
        print(f"Delete history error: {str(e)}")
        return jsonify({"error": f"Failed to delete scan: {str(e)}"}), 500

@app.route('/uploads/scan_images/<filename>')
def serve_scan_image(filename):
    """Serve uploaded scan images"""
    return send_from_directory(SCAN_IMAGES_FOLDER, filename)

# ============== ANALYTICS API ENDPOINTS ==============

@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary_endpoint():
    """Get descriptive analytics summary for a user"""
    try:
        email = request.args.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        summary = get_analytics_summary(email)
        if summary is None:
            return jsonify({"error": "Failed to fetch analytics summary"}), 500
            
        return jsonify({
            "success": True,
            "summary": summary
        }), 200
    except Exception as e:
        print(f"Analytics summary error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/charts', methods=['GET'])
def get_analytics_charts_endpoint():
    """Get predictive analytics data (charts)"""
    try:
        email = request.args.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        charts = get_analytics_charts(email)
        if charts is None:
            return jsonify({"error": "Failed to fetch analytics charts"}), 500
            
        return jsonify({
            "success": True,
            "charts": charts
        }), 200
    except Exception as e:
        print(f"Analytics charts error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/reports', methods=['GET'])
def get_analytics_reports_endpoint():
    """Get prescriptive analytics (recent reports)"""
    try:
        email = request.args.get('email')
        limit = request.args.get('limit', 10, type=int)
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        reports = get_analytics_reports(email, limit)
            
        return jsonify({
            "success": True,
            "reports": reports
        }), 200
    except Exception as e:
        print(f"Analytics reports error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ============== CHAT API ENDPOINTS ==============

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    AI chatbot endpoint for agricultural assistance
    Provides context-aware advice based on disease detection
    """
    try:
        from datetime import datetime
        data = request.json
        user_message = data.get('message')
        language = data.get('language', 'en')
        context = data.get('context', {})
        
        print(f"\n[DEBUG] Chat request received:")
        print(f"  Message: {user_message[:50]}..." if user_message else "  Message: None")
        print(f"  Language: {language}")
        print(f"  Context: {context}")
        
        # Validate input
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Validate language
        valid_languages = ['en', 'hi', 'kn', 'te', 'ta', 'bn']
        if language not in valid_languages:
            language = 'en'
        
        # Use chat service
        try:
            chat_service = get_chat_service()
            print(f"[OK] Chat service initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize chat service: {str(e)}")
            return jsonify({
                "error": "Chat service not available",
                "message": "The AI service is not properly configured. Please check the GEMINI_API_KEY environment variable."
            }), 503
        
        # Get AI response
        result = chat_service.get_chat_response(
            user_message=user_message,
            context=context,
            language=language
        )
        
        if not result.get('success'):
            print(f"[WARNING] Chat service returned error: {result.get('response')}")
            return jsonify({
                "error": "Failed to get response",
                "message": result.get('response')
            }), 500
        
        print(f"[OK] Chat response generated successfully")
        return jsonify({
            "success": True,
            "response": result['response'],
            "language": result['language']
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Chat failed: {str(e)}"
        }), 500

@app.route('/api/chat/greeting', methods=['POST'])
def chat_greeting():
    """Get initial greeting message based on disease detection"""
    try:
        data = request.json
        context = data.get('context', {})
        language = data.get('language', 'en')
        
        print(f"[DEBUG] Greeting request - Context: {context}, Language: {language}")
        
        # Import and use chat service
        try:
            chat_service = get_chat_service()
            print(f"[OK] Chat service initialized for greeting")
        except Exception as e:
            print(f"[ERROR] Failed to initialize chat service: {str(e)}")
            return jsonify({"error": "Chat service not available"}), 503
        
        greeting = chat_service.get_initial_greeting(context, language)
        print(f"[OK] Greeting generated: {greeting[:50]}...")
        return jsonify({"success": True, "greeting": greeting, "language": language}), 200
        
    except Exception as e:
        print(f"[ERROR] Greeting error: {str(e)}")
        return jsonify({"error": "Failed to generate AI greeting"}), 500


if __name__ == '__main__':
    # Initialize database on startup
    try:
        init_db()
        print("[OK] Database initialized")
    except Exception as e:
        print(f"[WARNING] Could not initialize database: {str(e)}")
    
    # Load multi-model system on startup (with timeout handling)
    try:
        print("[INFO] Starting model loading in background...")
        init_models()
    except Exception as e:
        print(f"[WARNING] Could not load models: {str(e)}")
        print("[INFO] Server will start but predictions will fail until models are available.")
    
    print(f"\n[OK] Flask server starting on http://0.0.0.0:5000")
    print(f"[INFO] Chat endpoint: POST /api/chat")
    print(f"[INFO] Chat greeting: POST /api/chat/greeting\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
