import requests
import os

API_URL = "http://127.0.0.1:8000/upload"
FILE_PATH = "C:\\Users\\CHARANTR\\Downloads\\Hackathon\\Hackathon\\sample_usecase.txt"

def test_upload():
    with open(FILE_PATH, "rb") as f:
        files = {"file": (os.path.basename(FILE_PATH), f, "text/plain")}
        response = requests.post(API_URL, files=files)
        print(f"Status Code: {response.status_code}")
        try:
            print("Response JSON:")
            print(response.json())
        except Exception:
            print("Raw Response:")
            print(response.text)

if __name__ == "__main__":
    test_upload()
