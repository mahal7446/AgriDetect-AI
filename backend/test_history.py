import requests
import sys

BASE_URL = "http://localhost:5000"
TEST_EMAIL = "mahalingm0@gmail.com"

def test_history():
    print(f"\n--- Testing Scan History for {TEST_EMAIL} ---")
    try:
        response = requests.get(f"{BASE_URL}/api/history/get?email={TEST_EMAIL}")
        data = response.json()
        print(f"Status Code: {response.status_code}")
        if data.get('success'):
            print(f"Count: {data.get('count')}")
            if data.get('history'):
                print("First History Item (keys):", data.get('history')[0].keys())
        else:
            print("Error:", data.get('error'))
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_history()
