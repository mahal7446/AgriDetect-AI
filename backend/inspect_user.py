from database import get_db_connection

def inspect_user():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Find user0
        cursor.execute("SELECT email, full_name, address FROM users WHERE email LIKE '%user0%' OR full_name LIKE '%user0%'")
        users = cursor.fetchall()
        print("Matching users:")
        for u in users:
            print(f"Email: {u['email']}, Name: {u['full_name']}, Address: {u['address']}")
            
        # Also check current logged in users (most recent)
        cursor.execute("SELECT email, full_name, address FROM users ORDER BY created_at DESC LIMIT 5")
        users = cursor.fetchall()
        print("\nRecent users:")
        for u in users:
            print(f"Email: {u['email']}, Name: {u['full_name']}, Address: {u['address']}")

if __name__ == "__main__":
    inspect_user()
