"""
Script to append new functions to database.py
"""

new_functions = '''
# ============== LOCATION-BASED AND NOTIFICATION FUNCTIONS ==============

from location_helpers import extract_district, normalize_district_name

def get_alerts_by_location(user_email, limit=20):
    """
    Get alerts filtered by user's location (same district or nearby)
    Returns: list of alerts
    """
    try:
        # Get user's address
        user = get_user(user_email)
        if not user or not user.get('address'):
            # If no address, return recent alerts
            return get_recent_alerts(limit)
        
        user_district = extract_district(user['address'])
        if not user_district:
            return get_recent_alerts(limit)
        
        # Normalize the district name
        normalized_district = normalize_district_name(user_district)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(\'\'\'
                SELECT id, farmer_name, location, disease_reported, description, prevention_methods, image_url, user_email, created_at
                FROM alerts
                ORDER BY created_at DESC
            \'\'\')
            
            rows = cursor.fetchall()
            filtered_alerts = []
            
            for row in rows:
                alert_location = row['location']
                alert_district = extract_district(alert_location)
                
                if alert_district:
                    normalized_alert_district = normalize_district_name(alert_district)
                    
                    # Match if districts are the same (after normalization)
                    # Or if district name appears in the location string
                    if (normalized_district == normalized_alert_district or 
                        user_district in alert_location.lower() or 
                        alert_district in user.get('address', '').lower()):
                        
                        filtered_alerts.append({
                            'id': row['id'],
                            'farmerName': row['farmer_name'],
                            'location': row['location'],
                            'diseaseReported': row['disease_reported'],
                            'description': row['description'],
                            'preventionMethods': row.get('prevention_methods'),
                            'imageUrl': row['image_url'],
                            'userEmail': row.get('user_email'),
                            'createdAt': row['created_at']
                        })
                        
                        if len(filtered_alerts) >= limit:
                            break
            
            return filtered_alerts
            
    except Exception as e:
        print(f"Error fetching location-based alerts: {str(e)}")
        return []

def delete_alert(alert_id, user_email):
    """
    Delete an alert (only if user owns it)
    Returns: (success, message_or_image_url)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # First check if alert exists and user owns it
            cursor.execute(\'\'\'
                SELECT image_url FROM alerts
                WHERE id = ? AND user_email = ?
            \'\'\', (alert_id, user_email))
            
            row = cursor.fetchone()
            if not row:
                return False, "Alert not found or not authorized"
            
            image_url = row['image_url'] if row else None
            
            # Delete the alert
            cursor.execute(\'\'\'
                DELETE FROM alerts
                WHERE id = ? AND user_email = ?
            \'\'\', (alert_id, user_email))
            
            if cursor.rowcount == 0:
                return False, "Alert not found"
            
        return True, image_url  # Return image URL for cleanup
    except Exception as e:
        return False, f"Error deleting alert: {str(e)}"

def get_user_notification_preference(email):
    """
    Get user's notification preference
    Returns: (success, enabled_boolean)
    """
    try:
        user = get_user(email)
        if not user:
            return False, False
        
        # Default to True if column doesn't exist or is None
        enabled = user.get('notifications_enabled', 1)
        return True, bool(enabled)
    except Exception as e:
        print(f"Error getting notification preference: {str(e)}")
        return False, True  # Default to enabled

def update_user_notification_preference(email, enabled):
    """
    Update user's notification preference
    Returns: (success, message)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(\'\'\'
                UPDATE users SET notifications_enabled = ?
                WHERE email = ?
            \'\'\', (1 if enabled else 0, email))
            
            if cursor.rowcount == 0:
                return False, "User not found"
            
        return True, "Notification preference updated successfully"
    except Exception as e:
        return False, f"Error updating notification preference: {str(e)}"

def get_new_alerts_count(user_email, last_seen_id=0):
    """
    Get count of new alerts since last_seen_id for user's location
    Returns: count of new alerts
    """
    try:
        alerts = get_alerts_by_location(user_email, limit=50)
        new_count = sum(1 for alert in alerts if alert['id'] > last_seen_id)
        return new_count
    except Exception as e:
        print(f"Error counting new alerts: {str(e)}")
        return 0
'''

# Append to database.py
with open('c:\\Users\\mahal\\Agri-Trail\\backend\\database.py', 'a', encoding='utf-8') as f:
    f.write(new_functions)

print("New functions added to database.py successfully!")
