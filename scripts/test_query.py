import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import json

s = requests.Session()
login = s.post("http://127.0.0.1:5000/api/login", json={"email": "test@test.com", "password":"pass"})
print("Login:", login.status_code)

res = s.post("http://127.0.0.1:5000/api/chat", json={"question": "What is IISF?", "stream": False})
print("Chat Response:")
print(json.dumps(res.json(), indent=2))
