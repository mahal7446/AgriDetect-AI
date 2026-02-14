"""
Test the health endpoint to check if backend is running
"""
import requests

def test_health():
    """Test the health endpoint"""
    url = "http://localhost:5000/health"
    
    try:
        print(f"Testing health endpoint: {url}")
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print("\n[SUCCESS] Backend is healthy!")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Model loaded: {result.get('model_loaded', False)}")
        else:
            print(f"\n[ERROR] Status code {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to backend server.")
        print("   Make sure the backend is running: python app.py")
    except Exception as e:
        print(f"[ERROR] {str(e)}")

if __name__ == '__main__':
    test_health()
