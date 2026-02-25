from database import get_alerts_by_location, get_db_connection

def final_verification():
    print("Running final verification for strict location isolation...")
    
    # User 0 (Bidar)
    email0 = "mahalingm0@gmail.com"
    # User 1 (Bengaluru Rural)
    email1 = "adm@gmail.com"
    
    print(f"\nFetching alerts for user0 (Bidar): {email0}")
    alerts0 = get_alerts_by_location(email0)
    print(f"User0 saw {len(alerts0)} alerts.")
    for a in alerts0:
        print(f" - ID: {a['id']}, Location: {a['location']}, User: {a['userEmail']}")
        if "bidar" not in a['location'].lower() and a['userEmail'] != email0:
            print("FAILURE: User0 (Bidar) saw a non-Bidar alert!")
            return
            
    print(f"\nFetching alerts for user1 (Bengaluru Rural): {email1}")
    alerts1 = get_alerts_by_location(email1)
    print(f"User1 saw {len(alerts1)} alerts.")
    for a in alerts1:
        print(f" - ID: {a['id']}, Location: {a['location']}, User: {a['userEmail']}")
        if "bengaluru" not in a['location'].lower() and "bangalore" not in a['location'].lower() and a['userEmail'] != email1:
            print("FAILURE: User1 (Bengaluru) saw a non-Bengaluru alert!")
            return

    print("\nSUCCESS: Strict location isolation confirmed!")

if __name__ == "__main__":
    final_verification()
