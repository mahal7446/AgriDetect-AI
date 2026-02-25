import sqlite3

DATABASE_NAME = 'backend/agridetect.db'

def query_db():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("\n--- Latest 10 Scan Records ---")
    cursor.execute("SELECT id, user_email, disease_name, crop_name, risk_level, health_status, scan_date FROM scan_history ORDER BY id DESC LIMIT 10")
    scans = cursor.fetchall()
    for row in scans:
        print(f"ID: {row['id']}, Email: '{row['user_email']}', Disease: {row['disease_name']}, Risk: {row['risk_level']}, Health: {row['health_status']}, Date: {row['scan_date']}")
        
    conn.close()

if __name__ == "__main__":
    query_db()
