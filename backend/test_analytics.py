import requests
import sys

BASE_URL = "http://localhost:5000"
TEST_EMAIL = "adm@gmail.com" # Should be a user that exists or has scans

def test_summary():
    print(f"\n--- Testing Analytics Summary for {TEST_EMAIL} ---")
    try:
        response = requests.get(f"{BASE_URL}/api/analytics/summary?email={TEST_EMAIL}")
        data = response.json()
        print(f"Status Code: {response.status_code}")
        if data.get('success'):
            print("Summary Data:", data.get('summary'))
        else:
            print("Error:", data.get('error'))
    except Exception as e:
        print(f"Request failed: {e}")

def test_charts():
    print(f"\n--- Testing Analytics Charts for {TEST_EMAIL} ---")
    try:
        response = requests.get(f"{BASE_URL}/api/analytics/charts?email={TEST_EMAIL}")
        data = response.json()
        print(f"Status Code: {response.status_code}")
        if data.get('success'):
            print("Charts Data (Yield):", data.get('charts', {}).get('yieldForecast'))
            print("Charts Data (Trends):", data.get('charts', {}).get('diseaseTrends'))
        else:
            print("Error:", data.get('error'))
    except Exception as e:
        print(f"Request failed: {e}")

def test_reports():
    print(f"\n--- Testing Analytics Reports for {TEST_EMAIL} ---")
    try:
        response = requests.get(f"{BASE_URL}/api/analytics/reports?email={TEST_EMAIL}")
        data = response.json()
        print(f"Status Code: {response.status_code}")
        if data.get('success'):
            print("Reports Count:", len(data.get('reports', [])))
            if data.get('reports'):
                print("First Report:", data.get('reports')[0])
        else:
            print("Error:", data.get('error'))
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # Check if backend is running first
    try:
        requests.get(BASE_URL)
    except:
        print(f"Error: Backend is not running at {BASE_URL}")
        sys.exit(1)
        
    test_summary()
    test_charts()
    test_reports()
