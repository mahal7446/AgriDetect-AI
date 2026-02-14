"""
SQLite database module for AgriDetect AI
Handles user authentication and scan history persistence
"""
import sqlite3
import bcrypt
from datetime import datetime
from contextlib import contextmanager

def _to_utc_iso(sqlite_timestamp):
    """Convert SQLite CURRENT_TIMESTAMP (UTC, no timezone info) to ISO 8601 with Z suffix.
    SQLite stores '2026-02-14 07:48:39' but JS needs '2026-02-14T07:48:39Z' to correctly
    interpret as UTC and convert to the user's local timezone."""
    if not sqlite_timestamp:
        return sqlite_timestamp
    # Replace space with T and append Z for UTC
    return sqlite_timestamp.replace(' ', 'T') + 'Z'

DATABASE_NAME = 'agridetect.db'

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_db():
    """Initialize database and create tables if they don't exist"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                profile_picture_url TEXT,
                address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migrate existing table if needed (add new columns)
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN profile_picture_url TEXT")
            print("Added profile_picture_url column")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN address TEXT")
            print("Added address column")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN notifications_enabled INTEGER DEFAULT 1")
            print("Added notifications_enabled column")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Create scan_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                crop_name TEXT,
                severity TEXT,
                risk_level TEXT,
                health_status TEXT,
                image_url TEXT,
                scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_email) REFERENCES users(email)
            )
        ''')
        
        # Migrate existing table if needed (add new columns)
        try:
            cursor.execute("ALTER TABLE scan_history ADD COLUMN image_url TEXT")
            print("Added image_url column to scan_history")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE scan_history ADD COLUMN risk_level TEXT")
            print("Added risk_level column to scan_history")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE scan_history ADD COLUMN health_status TEXT")
            print("Added health_status column to scan_history")
        except sqlite3.OperationalError:
            pass
        
        # Create community alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_name TEXT NOT NULL,
                location TEXT NOT NULL,
                disease_reported TEXT NOT NULL,
                description TEXT,
                prevention_methods TEXT,
                image_url TEXT,
                user_email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_email) REFERENCES users(email)
            )
        ''')
        
        # Migrate existing alerts table if needed
        try:
            cursor.execute("ALTER TABLE alerts ADD COLUMN prevention_methods TEXT")
            print("Added prevention_methods column to alerts")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        print("Database initialized successfully")

def hash_password(password):
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(stored_hash, provided_password):
    """Verify a password against its hash"""
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_hash.encode('utf-8'))

def create_user(email, full_name, phone, password):
    """
    Create a new user with hashed password
    Returns: (success, message_or_user_data)
    """
    try:
        password_hash = hash_password(password)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (email, full_name, phone, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (email, full_name, phone, password_hash))
            
        return True, {
            'email': email,
            'fullName': full_name,
            'phone': phone
        }
    except sqlite3.IntegrityError:
        return False, "Email already registered"
    except Exception as e:
        return False, f"Error creating user: {str(e)}"

def get_user(email):
    """
    Get user by email
    Returns: user dict or None
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, email, full_name, phone, password_hash, profile_picture_url, address, created_at
                FROM users WHERE email = ?
            ''', (email,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'email': row['email'],
                    'fullName': row['full_name'],
                    'phone': row['phone'],
                    'password_hash': row['password_hash'],
                    'profilePictureUrl': row['profile_picture_url'],
                    'address': row['address'],
                    'created_at': row['created_at']
                }
        return None
    except Exception as e:
        print(f"Error fetching user: {str(e)}")
        return None

def save_scan(user_email, disease_name, confidence, crop_name=None, severity=None, image_url=None, risk_level=None, health_status=None):
    """
    Save a disease detection scan to history
    Returns: (success, scan_id_or_message)
    """
    try:
        # Derive health_status and risk_level if not provided
        if health_status is None:
            health_status = 'Healthy' if 'healthy' in disease_name.lower() else 'Diseased'
        
        if risk_level is None:
            if health_status == 'Healthy':
                risk_level = 'Safe'
            elif confidence > 85:
                risk_level = 'High'
            else:
                risk_level = 'Medium'

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO scan_history (user_email, disease_name, confidence, crop_name, severity, image_url, risk_level, health_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_email, disease_name, confidence, crop_name, severity, image_url, risk_level, health_status))
            scan_id = cursor.lastrowid
            
        return True, scan_id
    except Exception as e:
        return False, f"Error saving scan: {str(e)}"

def get_user_scans(user_email, limit=50):
    """
    Get user's scan history
    Returns: list of scans
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT user_email FROM scan_history")
            print(f"DEBUG: Emails in DB: {[r[0] for r in cursor.fetchall()]}")
            print(f"DEBUG: Querying for email: '{user_email}'")
            
            cursor.execute('''
                SELECT id, disease_name, confidence, crop_name, severity, risk_level, health_status, image_url, scan_date
                FROM scan_history
                WHERE user_email = ?
                ORDER BY scan_date DESC
                LIMIT ?
            ''', (user_email, limit))
            
            rows = cursor.fetchall()
            return [{
                'id': row['id'],
                'diseaseName': row['disease_name'],
                'confidence': row['confidence'],
                'cropName': row['crop_name'],
                'severity': row['severity'],
                'riskLevel': row['risk_level'],
                'healthStatus': row['health_status'],
                'imageUrl': row['image_url'],
                'scanDate': _to_utc_iso(row['scan_date'])
            } for row in rows]
    except Exception as e:
        import traceback
        print(f"Error fetching scans for {user_email}: {str(e)}")
        traceback.print_exc()
        return []

def get_scan_by_id(scan_id, user_email):
    """
    Get a specific scan by ID (verifying ownership)
    Returns: scan dict or None
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, disease_name, confidence, crop_name, severity, image_url, scan_date
                FROM scan_history
                WHERE id = ? AND user_email = ?
            ''', (scan_id, user_email))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'diseaseName': row['disease_name'],
                    'confidence': row['confidence'],
                    'cropName': row['crop_name'],
                    'severity': row['severity'],
                    'imageUrl': row['image_url'],
                    'scanDate': _to_utc_iso(row['scan_date'])
                }
        return None
    except Exception as e:
        print(f"Error fetching scan: {str(e)}")
        return None

def delete_scan(scan_id, user_email):
    """
    Delete a scan from history (only if owned by user)
    Returns: (success, message_or_image_url)
    """
    try:
        # First get the scan to retrieve image URL for cleanup
        scan = get_scan_by_id(scan_id, user_email)
        if not scan:
            return False, "Scan not found or not authorized"
        
        image_url = scan.get('imageUrl')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM scan_history
                WHERE id = ? AND user_email = ?
            ''', (scan_id, user_email))
            
            if cursor.rowcount == 0:
                return False, "Scan not found"
            
        return True, image_url  # Return image URL for file cleanup
    except Exception as e:
        return False, f"Error deleting scan: {str(e)}"

def update_user_profile(email, full_name=None, phone=None, address=None):
    """
    Update user profile information
    Returns: (success, message)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build dynamic update query based on provided fields
            updates = []
            params = []
            
            if full_name is not None:
                updates.append("full_name = ?")
                params.append(full_name)
            if phone is not None:
                updates.append("phone = ?")
                params.append(phone)
            if address is not None:
                updates.append("address = ?")
                params.append(address)
            
            if not updates:
                return False, "No fields to update"
            
            params.append(email)
            query = f"UPDATE users SET {', '.join(updates)} WHERE email = ?"
            
            cursor.execute(query, params)
            
            if cursor.rowcount == 0:
                return False, "User not found"
            
        return True, "Profile updated successfully"
    except Exception as e:
        return False, f"Error updating profile: {str(e)}"

def update_profile_picture(email, profile_picture_url):
    """
    Update user profile picture URL
    Returns: (success, message)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET profile_picture_url = ?
                WHERE email = ?
            ''', (profile_picture_url, email))
            
            if cursor.rowcount == 0:
                return False, "User not found"
            
        return True, "Profile picture updated successfully"
    except Exception as e:
        return False, f"Error updating profile picture: {str(e)}"

def create_alert(farmer_name, location, disease_reported, description=None, prevention_methods=None, image_url=None, user_email=None):
    """
    Create a new community alert
    Returns: (success, message_or_alert_data)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (farmer_name, location, disease_reported, description, prevention_methods, image_url, user_email)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (farmer_name, location, disease_reported, description, prevention_methods, image_url, user_email))
            
            alert_id = cursor.lastrowid
            
        return True, {
            'id': alert_id,
            'farmerName': farmer_name,
            'location': location,
            'diseaseReported': disease_reported,
            'description': description,
            'preventionMethods': prevention_methods,
            'imageUrl': image_url
        }
    except Exception as e:
        return False, f"Error creating alert: {str(e)}"

def get_recent_alerts(limit=10):
    """
    Get recent community alerts
    Returns: list of alerts
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, farmer_name, location, disease_reported, description, prevention_methods, image_url, user_email, created_at
                FROM alerts
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [{
                'id': row['id'],
                'farmerName': row['farmer_name'],
                'location': row['location'],
                'diseaseReported': row['disease_reported'],
                'description': row['description'],
                'preventionMethods': row['prevention_methods'],
                'imageUrl': row['image_url'],
                'userEmail': row['user_email'],
                'createdAt': row['created_at']
            } for row in rows]
    except Exception as e:
        print(f"Error fetching alerts: {str(e)}")
        return []

# ============== LOCATION-BASED AND NOTIFICATION FUNCTIONS ==============

from location_helpers import extract_district, normalize_district_name

def get_alerts_by_location(user_email, limit=20):
    """
    Get alerts filtered by user's location (same district or nearby)
    Always includes the user's own alerts.
    Returns: list of alerts
    """
    try:
        # Get user's address
        user = get_user(user_email)
        user_address = user.get('address', '') if user else ''
        user_district = extract_district(user_address)
        normalized_user_district = normalize_district_name(user_district) if user_district else None
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Fetch more than limit to allow for filtering
            cursor.execute('''
                SELECT id, farmer_name, location, disease_reported, description, prevention_methods, image_url, user_email, created_at
                FROM alerts
                ORDER BY created_at DESC
                LIMIT 100
            ''')
            
            rows = cursor.fetchall()
            filtered_alerts = []
            
            for row in rows:
                alert_user_email = row['user_email']
                alert_location = row['location']
                
                # 1. Always include user's own alerts
                if alert_user_email == user_email:
                    filtered_alerts.append({
                        'id': row['id'],
                        'farmerName': row['farmer_name'],
                        'location': row['location'],
                        'diseaseReported': row['disease_reported'],
                        'description': row['description'],
                        'preventionMethods': row['prevention_methods'],
                        'imageUrl': row['image_url'],
                        'userEmail': row['user_email'],
                        'createdAt': row['created_at']
                    })
                    continue

                # 2. Location-based matching
                if normalized_user_district:
                    alert_district = extract_district(alert_location)
                    normalized_alert_district = normalize_district_name(alert_district) if alert_district else None
                    
                    # Match if districts are the same (after normalization)
                    if (normalized_user_district == normalized_alert_district or 
                        (user_district and user_district in alert_location.lower()) or 
                        (alert_district and alert_district in user_address.lower())):
                        
                        filtered_alerts.append({
                            'id': row['id'],
                            'farmerName': row['farmer_name'],
                            'location': row['location'],
                            'diseaseReported': row['disease_reported'],
                            'description': row['description'],
                            'preventionMethods': row['prevention_methods'],
                            'imageUrl': row['image_url'],
                            'userEmail': row['user_email'],
                            'createdAt': row['created_at']
                        })
                
                if len(filtered_alerts) >= limit:
                    break
            
            # 3. Fallback: if still no alerts (or very few), pad with recent ones
            if len(filtered_alerts) < 5:
                already_included = {a['id'] for a in filtered_alerts}
                for row in rows:
                    if row['id'] not in already_included:
                        filtered_alerts.append({
                            'id': row['id'],
                            'farmerName': row['farmer_name'],
                            'location': row['location'],
                            'diseaseReported': row['disease_reported'],
                            'description': row['description'],
                            'preventionMethods': row['prevention_methods'],
                            'imageUrl': row['image_url'],
                            'userEmail': row['user_email'],
                            'createdAt': row['created_at']
                        })
                    if len(filtered_alerts) >= limit:
                        break

            return filtered_alerts
            
    except Exception as e:
        print(f"Error fetching location-based alerts: {str(e)}")
        return get_recent_alerts(limit)

def delete_alert(alert_id, user_email):
    """
    Delete an alert (only if user owns it)
    Returns: (success, message_or_image_url)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # First check if alert exists and user owns it
            cursor.execute('''
                SELECT image_url FROM alerts
                WHERE id = ? AND user_email = ?
            ''', (alert_id, user_email))
            
            row = cursor.fetchone()
            if not row:
                return False, "Alert not found or not authorized"
            
            image_url = row['image_url'] if row else None
            
            # Delete the alert
            cursor.execute('''
                DELETE FROM alerts
                WHERE id = ? AND user_email = ?
            ''', (alert_id, user_email))
            
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
            cursor.execute('''
                UPDATE users SET notifications_enabled = ?
                WHERE email = ?
            ''', (1 if enabled else 0, email))
            
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

def update_alert(alert_id, user_email, farmer_name=None, location=None, disease_reported=None, description=None, prevention_methods=None, image_url=None):
    """
    Update an alert (only if user owns it)
    Returns: (success, message)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build dynamic update query
            updates = []
            params = []
            
            if farmer_name is not None:
                updates.append("farmer_name = ?")
                params.append(farmer_name)
            if location is not None:
                updates.append("location = ?")
                params.append(location)
            if disease_reported is not None:
                updates.append("disease_reported = ?")
                params.append(disease_reported)
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if prevention_methods is not None:
                updates.append("prevention_methods = ?")
                params.append(prevention_methods)
            if image_url is not None:
                updates.append("image_url = ?")
                params.append(image_url)
                
            if not updates:
                return False, "No fields to update"
                
            params.append(alert_id)
            params.append(user_email)
            
            query = f"UPDATE alerts SET {', '.join(updates)} WHERE id = ? AND user_email = ?"
            cursor.execute(query, params)
            
            if cursor.rowcount == 0:
                return False, "Alert not found or not authorized"
                
        return True, "Alert updated successfully"
    except Exception as e:
        return False, f"Error updating alert: {str(e)}"

def get_user_stats(user_email):
    """
    Get scan statistics for a user
    Returns: {total, healthy, diseased}
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get total scans
            cursor.execute('SELECT COUNT(*) FROM scan_history WHERE user_email = ?', (user_email,))
            total_scans = cursor.fetchone()[0]
            
            # Get healthy scans (where disease_name contains 'Healthy')
            cursor.execute('''
                SELECT COUNT(*) FROM scan_history 
                WHERE user_email = ? AND disease_name LIKE '%Healthy%'
            ''', (user_email,))
            healthy_scans = cursor.fetchone()[0]
            
            diseased_scans = total_scans - healthy_scans
            
            return {
                'total': total_scans,
                'healthy': healthy_scans,
                'diseased': diseased_scans
            }
    except Exception as e:
        print(f"Error fetching user stats: {str(e)}")
        return {'total': 0, 'healthy': 0, 'diseased': 0}

def get_total_users_count():
    """Get total number of registered users"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users')
            return {'success': True, 'total_users': cursor.fetchone()[0]}
    except Exception as e:
        print(f"Error fetching total users count: {str(e)}")
        return {'success': False, 'total_users': 0, 'error': str(e)}

def get_user_accuracy(user_email):
    """
    Get user's average accuracy (average confidence across all scans)
    Returns: average accuracy (0-100)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT AVG(confidence) FROM scan_history WHERE user_email = ?', (user_email,))
            avg_confidence = cursor.fetchone()[0]
            
            if avg_confidence is None:
                return 0.0
            
            # Since confidence is already stored as a percentage (e.g., 98.78),
            # we just return the average as is.
            return avg_confidence
    except Exception as e:
        print(f"Error fetching user accuracy: {str(e)}")
        return 0.0

def get_analytics_summary(user_email):
    """
    Get descriptive analytics summary for a user
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total Scans
            cursor.execute('SELECT COUNT(*) FROM scan_history WHERE user_email = ?', (user_email,))
            total_scans = cursor.fetchone()[0]
            
            if total_scans == 0:
                return {
                    'totalScans': 0,
                    'averageHealth': 0,
                    'diseaseAlerts': 0,
                    'estimatedYield': 0
                }
            
            # Healthy vs Diseased
            cursor.execute('''
                SELECT COUNT(*) FROM scan_history 
                WHERE user_email = ? AND (health_status = 'Healthy' OR disease_name LIKE '%Healthy%')
            ''', (user_email,))
            healthy_count = cursor.fetchone()[0]
            
            avg_health = (healthy_count / total_scans) * 100
            
            # Disease Alerts (High Risk)
            cursor.execute('''
                SELECT COUNT(*) FROM scan_history 
                WHERE user_email = ? AND risk_level = 'High'
            ''', (user_email,))
            high_risk_count = cursor.fetchone()[0]
            
            # Estimated Yield (Derive from health)
            # Base yield is 100%, each diseased plant reduces it by some factor
            # Let's say a diseased plant reduces yield by 30% on average
            estimated_yield = max(0, 100 - ((total_scans - healthy_count) / total_scans * 30))
            
            return {
                'totalScans': total_scans,
                'averageHealth': round(avg_health, 1),
                'diseaseAlerts': high_risk_count,
                'estimatedYield': round(estimated_yield, 1)
            }
    except Exception as e:
        print(f"Error fetching analytics summary: {str(e)}")
        return None

def get_analytics_charts(user_email):
    """
    Get predictive analytics data (charts)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Crop Yield Forecast (Bar Chart)
            # Group by crop type and calculate health score as a proxy for yield
            cursor.execute('''
                SELECT crop_name, 
                       AVG(CASE WHEN health_status = 'Healthy' THEN 100 ELSE 60 END) as yield_score
                FROM scan_history
                WHERE user_email = ? AND crop_name IS NOT NULL
                GROUP BY crop_name
            ''', (user_email,))
            yield_data = [{'crop': row['crop_name'], 'yield': round(row['yield_score'], 1)} for row in cursor.fetchall()]
            
            # 2. Disease Trends (Line Chart - 6 Months)
            # Generate the last 6 months for complete coverage
            from datetime import datetime, timedelta
            import calendar
            now = datetime.utcnow()
            months_range = []
            for i in range(5, -1, -1):
                # Go back i months from now
                dt = now.replace(day=1) - timedelta(days=i * 30)
                ym = dt.strftime('%Y-%m')
                short_name = calendar.month_abbr[dt.month]
                months_range.append((ym, short_name))

            cursor.execute('''
                SELECT strftime('%Y-%m', scan_date) as month,
                       SUM(CASE WHEN health_status = 'Healthy' THEN 1 ELSE 0 END) as healthy,
                       SUM(CASE WHEN health_status = 'Diseased' THEN 1 ELSE 0 END) as diseased
                FROM scan_history
                WHERE user_email = ?
                  AND scan_date >= date('now', '-6 months')
                GROUP BY month
                ORDER BY month ASC
            ''', (user_email,))
            
            # Build a lookup from query results
            db_trends = {}
            for row in cursor.fetchall():
                db_trends[row['month']] = {'healthy': row['healthy'], 'diseased': row['diseased']}

            # Merge with full 6-month range so every month appears on the chart
            trends = []
            for ym, short_name in months_range:
                data = db_trends.get(ym, {'healthy': 0, 'diseased': 0})
                trends.append({
                    'month': short_name,
                    'healthy': data['healthy'],
                    'diseased': data['diseased']
                })
            
            # If no data, return some default structure
            if not yield_data:
                yield_data = [
                    {'crop': 'Rice', 'yield': 85},
                    {'crop': 'Wheat', 'yield': 92},
                    {'crop': 'Potato', 'yield': 78},
                    {'crop': 'Tomato', 'yield': 88}
                ]
            
            return {
                'yieldForecast': yield_data,
                'diseaseTrends': trends
            }
    except Exception as e:
        print(f"Error fetching analytics charts: {str(e)}")
        return None

def get_analytics_reports(user_email, limit=10):
    """
    Get prescriptive analytics (recent reports)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, scan_date, crop_name, disease_name, risk_level
                FROM scan_history
                WHERE user_email = ?
                ORDER BY scan_date DESC
                LIMIT ?
            ''', (user_email, limit))
            
            return [{
                'id': row['id'],
                'date': _to_utc_iso(row['scan_date']),
                'cropType': row['crop_name'] or 'Unknown',
                'diagnosis': row['disease_name'],
                'riskLevel': row['risk_level'] or 'Medium'
            } for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching analytics reports: {str(e)}")
        return []
