import streamlit as st
import os
from dotenv import load_dotenv
from bedrock_agentcore.memory import MemoryClient
from flight_booking_agent import create_flight_booking_agent
from hotel_booking_agent import create_hotel_booking_agent, extract_cities_with_claude
from activity_planner_agent import create_activity_planner_agent
import concurrent.futures

# --- Load Snowflake credentials from .env file ---
load_dotenv()

# --- AWS AgentCore memory setup ---
REGION = os.getenv('AWS_REGION', 'us-east-1')
MEMORY_CLIENT = MemoryClient(region_name=REGION)
MEMORY_NAME = f"FlightAgentMemory"
try:
    memory = MEMORY_CLIENT.create_memory_and_wait(
        name=MEMORY_NAME,
        strategies=[],
        description="Short-term memory for flight booking agent",
        event_expiry_days=7,
    )
    MEMORY_ID = memory['id']
except Exception:
    memories = MEMORY_CLIENT.list_memories()
    MEMORY_ID = next((m['id'] for m in memories if m['id'].startswith(MEMORY_NAME)), None)

ACTOR_ID = "flight_user_001"
SESSION_ID = "flight_session_001"

flight_agent = create_flight_booking_agent(MEMORY_CLIENT, MEMORY_ID, ACTOR_ID, SESSION_ID, region=REGION)
hotel_agent = create_hotel_booking_agent(MEMORY_CLIENT, MEMORY_ID, ACTOR_ID, SESSION_ID, region=REGION)
activity_agent = create_activity_planner_agent(MEMORY_CLIENT, MEMORY_ID, ACTOR_ID, SESSION_ID, region=REGION)


def main():
    st.title("Travel Booking Agent (Cortex + AgentCore)")
    st.markdown("Enter your travel request below. The agent will use AWS AgentCore for memory and Cortex Analyst for SQL generation and execution. Hotels will be searched for all destination cities mentioned in your query.")

    travel_query = st.text_input("Enter your travel request (e.g., 'delhi to mumbai and pune and back'):")
    if travel_query:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            flight_future = executor.submit(flight_agent, travel_query)
            hotel_cities = extract_cities_with_claude(travel_query, region=REGION)
            hotel_futures = [executor.submit(hotel_agent, f"Find hotels in {city}") for city in hotel_cities]
            activity_future = executor.submit(activity_agent, travel_query)

            st.header("Flight Booking")
            with st.spinner("Processing your flight booking request..."):
                streamed_text, interpretation, sql_code, best_option, df = flight_future.result()
                if streamed_text:
                    st.markdown(streamed_text.replace("•", "\n\n"))
                if interpretation:
                    st.markdown(f"**Interpretation:** {interpretation}")
                if best_option:
                    st.markdown("---")
                    st.markdown("### Best Flight Option (Claude):")
                    st.markdown(best_option.strip())
                if df is not None and not df.empty:
                    with st.expander("Show all flight options"):
                        st.dataframe(df)
                if sql_code:
                    with st.expander("Show generated SQL"):
                        st.code(sql_code, language="sql")
                elif df is not None:
                    st.info("No results found.")
                elif sql_code is None:
                    st.error("Error running SQL or generating results.")

            st.header("Hotel Booking (for all destination cities)")
            for city, future in zip(hotel_cities, hotel_futures):
                streamed_text, interpretation, sql_code, best_option, df = future.result()
                st.subheader(f"Hotels in {city}")
                if streamed_text:
                    st.markdown(streamed_text.replace("•", "\n\n"))
                if interpretation:
                    st.markdown(f"**Interpretation:** {interpretation}")
                if best_option:
                    st.markdown("---")
                    st.markdown(f"### Best Hotel Option in {city} (Claude):")
                    st.markdown(best_option.strip())
                if df is not None and not df.empty:
                    with st.expander(f"Show all hotel options for {city}"):
                        st.dataframe(df)
                if sql_code:
                    with st.expander(f"Show generated SQL for {city}"):
                        st.code(sql_code, language="sql")
                elif df is not None:
                    st.info("No results found.")
                elif sql_code is None:
                    st.error("Error running SQL or generating results.")

            st.header("Activity Planner (Cortex Search)")
            with st.spinner("Searching for activities..."):
                streamed_text, interpretation, summary_md, activity_df = activity_future.result()
                st.markdown(summary_md)
                if activity_df is not None and not activity_df.empty:
                    with st.expander("Show all activity search results"):
                        st.dataframe(activity_df)
                else:
                    st.info("No activities found.")

if __name__ == "__main__":
    main()
