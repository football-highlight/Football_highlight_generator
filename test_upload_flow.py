# test_upload_flow.py
import requests
import json

API_BASE_URL = "http://localhost:8000/api"

print("Testing complete upload → process flow...")

# 1. Check the latest uploaded file
print("\n1. Checking latest uploaded files...")
response = requests.get(f"{API_BASE_URL}/videos")
if response.status_code == 200:
    videos = response.json().get("videos", [])
    if videos:
        latest_video = videos[0]  # Most recent
        filename = latest_video["filename"]
        print(f"   Latest video: {filename}")
        print(f"   Size: {latest_video['size']} bytes")
        print(f"   Path: {latest_video['path']}")
        
        # 2. Try to process this file
        print(f"\n2. Trying to process: {filename}")
        process_data = {
            "filename": filename,
            "generate_highlights": True
        }
        
        print(f"   Sending data: {json.dumps(process_data)}")
        process_response = requests.post(f"{API_BASE_URL}/process", json=process_data)
        
        print(f"   Status: {process_response.status_code}")
        print(f"   Response: {process_response.text}")
        
        # 3. Check what the API sees
        print(f"\n3. Checking API health...")
        health_response = requests.get(f"{API_BASE_URL}/health")
        print(f"   Health: {health_response.json()}")
    else:
        print("   No videos found!")
else:
    print(f"   Error getting videos: {response.status_code} - {response.text}")