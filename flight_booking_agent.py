import os
from dotenv import load_dotenv
import snowflake.connector
from strands import Agent
from bedrock_agentcore.memory import MemoryClient
import requests
import json
import boto3

# Load environment variables
load_dotenv()
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
CORTEX_AGENT_URL = "https://sfseapac-bsuresh.snowflakecomputing.com/api/v2/cortex/agent:run"
AUTH_TOKEN = os.getenv("CORTEX_AUTH_TOKEN")
API_TIMEOUT = 60
SEMANTIC_MODELS = "@TRAVEL_DB.PUBLIC.DOCS/travel_data.yaml"

if not AUTH_TOKEN or len(AUTH_TOKEN) < 20:
    raise RuntimeError("CORTEX_AUTH_TOKEN is missing or too short. Please check your .env file and restart the app.")
else:
    print("Loaded Cortex token (first 10 chars):", AUTH_TOKEN[:10])

REGION = os.getenv('AWS_REGION', 'us-west-2')
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Short-term memory helper
def get_recent_context(memory_client, memory_id, actor_id, session_id, k=5):
    recent_turns = memory_client.get_last_k_turns(
        memory_id=memory_id,
        actor_id=actor_id,
        session_id=session_id,
        k=k
    )
    context = ""
    for turn in recent_turns:
        for message in turn:
            role = message['role']
            content = message['content']['text']
            context += f"{role.title()}: {content}\n"
    return context

def cortex_analyst_sql(query: str):
    payload = {
        "model": "llama3.1-70b",
        "response_instruction": "Generate SQL for flight booking queries.",
        "tools": [
            {
                "tool_spec": {
                    "type": "cortex_analyst_text_to_sql",
                    "name": "Analyst1"
                }
            }
        ],
        "tool_resources": {
            "Analyst1": {"semantic_model_file": SEMANTIC_MODELS}
        },
        "tool_choice": {"type": "tool", "name": ["Analyst1"]},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ]
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        print("Using Cortex token (first 10 chars):", AUTH_TOKEN[:10])
        resp = requests.post(CORTEX_AGENT_URL, headers=headers, json=payload, timeout=API_TIMEOUT)
        print("Cortex Analyst raw response:", resp.text)  # Debug print
        if resp.status_code != 200:
            return None, f"Cortex Analyst error: {resp.status_code} {resp.text}"
        data = resp.json()
        # Parse for SQL in the response
        sql = None
        interpretation = None
        if 'delta' in data:
            content = data['delta'].get('content', [])
        elif 'content' in data:
            content = data['content']
        else:
            content = []
        for content_item in content:
            if content_item.get('type') == 'tool_results':
                tool_results = content_item.get('tool_results', {})
                for result in tool_results.get('content', []):
                    if result.get('type') == 'json':
                        json_data = result.get('json', {})
                        sql = json_data.get('sql')
                        interpretation = json_data.get('text')
        return sql, interpretation
    except Exception as e:
        return None, f"Cortex Analyst exception: {e}"

def cortex_analyst_sql_stream(query: str):
    payload = {
        "model": "llama3.1-70b",
        "response_instruction": "Generate SQL for flight booking queries.",
        "tools": [
            {
                "tool_spec": {
                    "type": "cortex_analyst_text_to_sql",
                    "name": "Analyst1"
                }
            }
        ],
        "tool_resources": {
            "Analyst1": {"semantic_model_file": SEMANTIC_MODELS}
        },
        "tool_choice": {"type": "tool", "name": ["Analyst1"]},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ]
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        print("Using Cortex token (first 10 chars):", AUTH_TOKEN[:10])
        with requests.post(CORTEX_AGENT_URL, headers=headers, json=payload, timeout=API_TIMEOUT, stream=True) as resp:
            buffer = ""
            prev_event = None
            for line in resp.iter_lines(decode_unicode=True):
                if line is None or line.strip() == "":
                    continue
                if line.startswith("event:"):
                    prev_event = line.strip()
                elif line.startswith("data:"):
                    if prev_event == "event: message.delta":
                        data_json = line[len("data: "):]
                        try:
                            data = json.loads(data_json)
                            yield data
                        except Exception as e:
                            print(f"JSON decode error: {e}")
                    elif prev_event == "event: error":
                        print(f"API error event: {line}")
                elif line.startswith("event: done"):
                    break
    except Exception as e:
        yield {"error": f"Cortex Analyst streaming exception: {e}"}

def run_sql_on_snowflake(sql):
    try:
        ctx = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
        )
        cs = ctx.cursor()
        cs.execute(sql)
        df = cs.fetch_pandas_all()
        cs.close()
        ctx.close()
        return df
    except Exception as e:
        return None

def get_best_flight_from_claude(table_df, user_query, region="us-east-1"):
    table_str = table_df.to_markdown(index=False)
    system_prompt = (
        "You are a travel assistant. Given a table of flight options and a user query, "
        "return the best 3 flight options as english text with all details"
    )
    user_message = (
        f"The user asked: {user_query}\n"
        f"Here are the available flight options in a table:\n\n"
        f"{table_str}"
    )
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 1,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip() if result.get("content") else ""


def create_flight_booking_agent(memory_client, memory_id, actor_id, session_id, region="us-east-1"):
    def agent(query):
        # 1. Streamed response (collect for UI)
        streamed_text = ""
        interpretation = None
        sql_code = None
        for data in cortex_analyst_sql_stream(query):
            if "error" in data:
                return streamed_text, interpretation, None, None, None
            delta = data.get("delta", {})
            content = delta.get("content", [])
            for content_item in content:
                if content_item.get("type") == "text":
                    streamed_text += content_item.get("text", "")
                elif content_item.get("type") == "tool_results":
                    tool_results = content_item.get("tool_results", {})
                    for result in tool_results.get("content", []):
                        if result.get("type") == "json":
                            json_data = result.get("json", {})
                            if "text" in json_data:
                                interpretation = json_data["text"]
                            if "sql" in json_data:
                                sql_code = json_data["sql"]
        if not sql_code:
            return streamed_text, interpretation, None, None, None
        # 2. Run SQL on Snowflake
        df = run_sql_on_snowflake(sql_code)
        # 3. Ask Claude for best option if results exist
        best_option = None
        if df is not None and not df.empty:
            best_option = get_best_flight_from_claude(df, query, region=region)
        return streamed_text, interpretation, sql_code, best_option, df
    return agent
