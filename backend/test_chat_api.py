import requests
import json

def main():
    url = "http://127.0.0.1:5000/api/chat"
    payload = {
        "message": "How to treat leaf blight?",
        "language": "en",
        "context": {
            "crop": "Wheat",
            "disease": "Wheat_Leaf_Blight",
            "confidence": 0.9
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        print("Status:", resp.status_code)
        print("Body:", resp.text)
    except Exception as e:
        print("Request failed:", e)

if __name__ == "__main__":
    main()
