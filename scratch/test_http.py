import urllib.request
import json

resp_index = urllib.request.urlopen('http://127.0.0.1:5000/')
print(f"Index HTML Status: {resp_index.status}, bytes: {len(resp_index.read())}")

resp_status = urllib.request.urlopen('http://127.0.0.1:5000/api/status')
status_data = json.loads(resp_status.read().decode())
print(f"API /status: {status_data}")

resp_labels = urllib.request.urlopen('http://127.0.0.1:5000/api/labels')
labels_data = json.loads(resp_labels.read().decode())
print(f"API /labels: {len(labels_data)} labels loaded successfully.")
