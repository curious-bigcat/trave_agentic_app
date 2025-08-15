import os
from dotenv import load_dotenv
import snowflake.connector
from bedrock_agentcore.memory import MemoryClient
import requests
import boto3
import json

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
SEMANTIC_MODELS = "@TRAVEL_DB.PUBLIC.DOCS/hotel_booking.yaml"

REGION = os.getenv('AWS_REGION', 'us-east-1')
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Streaming Cortex Analyst SQL generation
def cortex_analyst_sql_stream(query: str):
    payload = {
        "model": "llama3.1-70b",
        "response_instruction": "Generate SQL for hotel booking queries.",
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
        with requests.post(CORTEX_AGENT_URL, headers=headers, json=payload, timeout=API_TIMEOUT, stream=True) as resp:
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

# Run SQL on Snowflake
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

# Bedrock Claude best option selection
def get_best_hotel_from_claude(table_df, user_query, region="us-east-1"):
    table_str = table_df.to_markdown(index=False)
    system_prompt = (
        "You are a travel assistant. Given a table of hotel options and a user query, "
        "return the best 3 flight options as english text with all details"
    )
    user_message = (
        f"The user asked: {user_query}\n"
        f"Here are the available hotel options in a table:\n\n"
        f"{table_str}"
    )
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
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

def extract_cities_with_claude(user_query, region="us-east-1"):
    system_prompt = (
        "You are a travel assistant. Given a user travel query, extract the destination cities. Skip the source city "
        "Return only a valid Python list of city names, e.g. ['Mumbai', 'Pune']. Do not explain."
    )
    user_message = f"Extract all destination cities from this travel query: \"{user_query}\"."
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 128,
        "temperature": 0.0,
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
    text = result["content"][0]["text"].strip() if result.get("content") else ""
    # Try to safely evaluate the list
    import ast
    try:
        cities = ast.literal_eval(text)
        if isinstance(cities, list):
            return [str(city) for city in cities]
    except Exception:
        pass
    return []

# Main agent factory
def create_hotel_booking_agent(memory_client, memory_id, actor_id, session_id, region="us-east-1"):
    def agent(query):
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
        df = run_sql_on_snowflake(sql_code)
        best_option = None
        if df is not None and not df.empty:
            best_option = get_best_hotel_from_claude(df, query, region=region)
        return streamed_text, interpretation, sql_code, best_option, df
    return agent
