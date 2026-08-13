import requests
import json
import sys

url = "http://localhost:8000/api/analyze/"
file_path = r"D:\deepfake_detect\data\video\val\real\id5_0001.mp4"

print(f"Testing API endpoint {url} with file {file_path}")

try:
    with open(file_path, "rb") as f:
        files = {"file": (file_path.split("\\")[-1], f, "application/octet-stream")}
        response = requests.post(url, files=files)

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success! Response JSON:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error Response:")
        print(response.text)
except requests.exceptions.ConnectionError:
    print("Connection error: Is the Django server running?")
    sys.exit(1)
except Exception as e:
    print(f"Exception occurred: {e}")
    sys.exit(1)
