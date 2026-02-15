from database import get_db_connection

def cleanup_and_inspect():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Delete test alerts
        emails_to_delete = ["test_user_blr@example.com", "test_user_bidar@example.com", "test_user_no_loc@example.com"]
        cursor.execute("DELETE FROM alerts WHERE user_email IN (?, ?, ?)", (emails_to_delete[0], emails_to_delete[1], emails_to_delete[2]))
        print(f"Deleted {cursor.rowcount} test alerts.")
        
        cursor.execute("DELETE FROM users WHERE email IN (?, ?, ?)", (emails_to_delete[0], emails_to_delete[1], emails_to_delete[2]))
        print(f"Deleted {cursor.rowcount} test users.")
        
        # Inspect remaining alerts
        cursor.execute("SELECT id, location, user_email FROM alerts")
        alerts = cursor.fetchall()
        print("\nExisting alerts in DB:")
        for a in alerts:
            print(f"ID: {a['id']}, Location: {a['location']}, User: {a['user_email']}")
            
        conn.commit()

if __name__ == "__main__":
    cleanup_and_inspect()
