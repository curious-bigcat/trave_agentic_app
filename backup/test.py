import requests

AUTH_TOKEN = "eyJraWQiOiIyMjU5Mjg1OTQyNTEzNjc0IiwiYWxnIjoiRVMyNTYifQ.eyJwIjoiMTM0NjYzOTQwOjM0NDczOTY3ODc3IiwiaXNzIjoiU0Y6MTA0MyIsImV4cCI6MTc1NjgxOTA5NX0.KVuBFMR7GEKSfgoIdZgCk0oGtNt-8FpX0h7SfYQyIiBISvELCki2VG_x9thUeiRxjtvkS3IEq_pla9_vtLATnw"
CORTEX_AGENT_URL = "https://sfseapac-bsuresh.snowflakecomputing.com/api/v2/cortex/agent:run"

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama3.1-70b",
    "response_instruction": "Test",
    "tools": [],
    "tool_resources": {},
    "tool_choice": {"type": "auto"},
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"}
            ]
        }
    ]
}

resp = requests.post(CORTEX_AGENT_URL, headers=headers, json=payload, timeout=50)
print(resp.status_code)
print(resp.text)