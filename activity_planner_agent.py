import os
from dotenv import load_dotenv
import pandas as pd
from snowflake.snowpark import Session
from snowflake.core import Root
import boto3
import json
import ast
from bedrock_agentcore.memory import MemoryClient

load_dotenv()

CONNECTION_PARAMETERS = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE", "test_role"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "test_warehouse"),
    "database": "TRAVEL_DB",
    "schema": "PUBLIC",
}
CORTEX_SEARCH_COLUMNS = ["CHUNK", "CATEGORY", "CHUNK_INDEX", "RELATIVE_PATH"]

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

def extract_trip_info_with_claude(user_query, region="us-east-1"):
    system_prompt = (
        "You are a travel assistant. Given a user travel query, extract the following as a Python dict: "
        "'source_city': the city the trip starts from, 'dest_cities': a list of destination cities (excluding the source), "
        "'duration': the number of days for the trip (as an integer, or None if not specified). "
        "Return only a valid Python dict, e.g. {'dest_cities': ['Mumbai', 'Pune'], 'duration': 15 days}. Do not explain."
    )
    user_message = f"Extract the source city, destination cities, and trip duration from this travel query: \"{user_query}\"."
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
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
    try:
        info = ast.literal_eval(text)
        if isinstance(info, dict):
            return info
    except Exception:
        pass
    return {"source_city": "", "dest_cities": [], "duration": None}

def get_daywise_plan_from_claude(activities_df, user_query, source_city, dest_cities, region="us-east-1"):
    table_str = activities_df.to_markdown(index=False)
    system_prompt = (
        "You are a travel assistant. Given a table of activities and a user query, create a detailed day-wise travel plan. "
        "The trip starts in the source city and visits only the destination cities. "
        "Do NOT include the source city plan. "
        "Create a detailed plan for each day, suggest the  activities, and present the plan as detailed as possible."
    )
    user_message = (
        f"The user asked: {user_query}\n"
        f"Source city: {source_city}\n"
        f"Destination cities: {', '.join(dest_cities)}\n"
        f"Here are the available activities in a table:\n\n"
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

def create_activity_planner_agent(memory_client, memory_id, actor_id, session_id, region="us-east-1"):
    def agent(user_query, limit=10):
        context = get_recent_context(memory_client, memory_id, actor_id, session_id)
        trip_info = extract_trip_info_with_claude(user_query, region=region)
        source_city = trip_info.get("source_city", "")
        dest_cities = trip_info.get("dest_cities", [])
        duration = trip_info.get("duration", None)
        dest_cities = [city for city in dest_cities if city.strip().lower() != source_city.strip().lower()]
        session = Session.builder.configs(CONNECTION_PARAMETERS).create()
        root = Root(session)
        my_service = (
            root
            .databases["TRAVEL_DB"]
            .schemas["PUBLIC"]
            .cortex_search_services["TRAVEL_SEARCH_SERVICE"]
        )
        all_activities = []
        for city in dest_cities:
            cortex_query = f"Find activities and experiences for a traveler in {city}. Trip duration: {duration} days."
            resp = my_service.search(
                query=cortex_query,
                columns=CORTEX_SEARCH_COLUMNS,
                limit=limit,
            )
            df = None
            if hasattr(resp, "to_df"):
                try:
                    df = resp.to_df()
                except Exception as e:
                    print(".to_df() failed:", e)
            if df is None and hasattr(resp, "results"):
                try:
                    df = pd.DataFrame(resp.results)
                except Exception as e:
                    print("pd.DataFrame(resp.results) failed:", e)
            if df is None:
                try:
                    df = pd.DataFrame(resp)
                except Exception as e:
                    print("pd.DataFrame(resp) failed:", e)
            if df is not None and not df.empty:
                df["CITY"] = city
                all_activities.append(df)
        if all_activities:
            activities_df = pd.concat(all_activities, ignore_index=True)
        else:
            activities_df = pd.DataFrame()
        if not activities_df.empty:
            plan_md = get_daywise_plan_from_claude(
                activities_df, user_query, source_city, dest_cities, region=region
            )
        else:
            plan_md = "No activities found."
        return "", "", plan_md, activities_df
    return agent
