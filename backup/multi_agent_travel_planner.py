import logging
from datetime import datetime
import os
from strands import Agent, tool
from strands.hooks import AgentInitializedEvent, HookProvider, HookRegistry, MessageAddedEvent
from bedrock_agentcore.memory import MemoryClient
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("multi-agent-travel")

# Configuration
REGION = os.getenv('AWS_REGION', 'us-west-2')
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Create shared memory resource
client = MemoryClient(region_name=REGION)
memory_name = f"TravelAgent_STM_{datetime.now().strftime('%Y%m%d%H%M%S')}"
memory_id = None
try:
    logger.info("Creating shared memory resource...")
    memory = client.create_memory_and_wait(
        name=memory_name,
        description="Travel Agent STM",
        strategies=[],
        event_expiry_days=7,
        max_wait=300,
        poll_interval=10
    )
    memory_id = memory['id']
    logger.info(f"✅ Memory created with ID: {memory_id}")
except ClientError as e:
    if e.response['Error']['Code'] == 'ValidationException' and "already exists" in str(e):
        memories = client.list_memories()
        memory_id = next((m['id'] for m in memories if m['id'].startswith(memory_name)), None)
        logger.info(f"Memory already exists. Using existing memory ID: {memory_id}")
    else:
        logger.error(f"❌ ERROR: {e}")
        raise
except Exception as e:
    logger.error(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    if memory_id:
        try:
            client.delete_memory_and_wait(memory_id=memory_id)
            logger.info(f"Cleaned up memory: {memory_id}")
        except Exception as cleanup_error:
            logger.error(f"Failed to clean up memory: {cleanup_error}")
    raise

# Short-term memory hook
class ShortTermMemoryHook(HookProvider):
    def __init__(self, memory_client: MemoryClient, memory_id: str, actor_id: str, session_id: str):
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
    def on_agent_initialized(self, event: AgentInitializedEvent):
        try:
            recent_turns = self.memory_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                k=5
            )
            if recent_turns:
                context_messages = []
                for turn in recent_turns:
                    for message in turn:
                        role = message['role']
                        content = message['content']['text']
                        context_messages.append(f"{role.title()}: {content}")
                context = "\n".join(context_messages)
                event.agent.system_prompt += f"\n\nRecent conversation history:\n{context}\n\nContinue the conversation naturally based on this context."
                logger.info(f"✅ Loaded {len(recent_turns)} recent conversation turns")
            else:
                logger.info("No previous conversation history found")
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
    def on_message_added(self, event: MessageAddedEvent):
        messages = event.agent.messages
        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(str(messages[-1].get("content", "")), messages[-1]["role"])]
            )
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
    def register_hooks(self, registry: HookRegistry):
        registry.add_callback(MessageAddedEvent, self.on_message_added)
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)

def main():
    # Unique actor/session IDs
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    flight_actor_id = f"flight-user-{now}"
    hotel_actor_id = f"hotel-user-{now}"
    activity_actor_id = f"activity-user-{now}"
    session_id = f"travel-session-{now}"

    # System prompts
    HOTEL_BOOKING_PROMPT = (
        "You are a hotel booking assistant. Help customers find hotels, make reservations, and answer questions about accommodations and amenities. "
        "Provide clear information about availability, pricing, and booking procedures in a friendly, helpful manner."
    )
    FLIGHT_BOOKING_PROMPT = (
        "You are a flight booking assistant. Help customers find flights, make reservations, and answer questions about airlines, routes, and travel policies. "
        "Provide clear information about flight availability, pricing, schedules, and booking procedures in a friendly, helpful manner."
    )
    ACTIVITY_PLANNER_PROMPT = (
        "You are an activity planner assistant. Help customers plan activities, tours, and experiences at their travel destination. "
        "Suggest local attractions, events, and things to do based on user interests, dates, and location."
    )

    # Specialized agent tools
    @tool
    def flight_booking_assistant(query: str) -> str:
        try:
            flight_memory_hooks = ShortTermMemoryHook(
                memory_client=client,
                memory_id=memory_id,
                actor_id=flight_actor_id,
                session_id=session_id
            )
            flight_agent = Agent(hooks=[flight_memory_hooks], model=MODEL_ID, system_prompt=FLIGHT_BOOKING_PROMPT)
            response = flight_agent(query)
            return str(response)
        except Exception as e:
            return f"Error in flight booking assistant: {str(e)}"

    @tool
    def hotel_booking_assistant(query: str) -> str:
        try:
            hotel_memory_hooks = ShortTermMemoryHook(
                memory_client=client,
                memory_id=memory_id,
                actor_id=hotel_actor_id,
                session_id=session_id
            )
            hotel_agent = Agent(hooks=[hotel_memory_hooks], model=MODEL_ID, system_prompt=HOTEL_BOOKING_PROMPT)
            response = hotel_agent(query)
            return str(response)
        except Exception as e:
            return f"Error in hotel booking assistant: {str(e)}"

    @tool
    def activity_planner_assistant(query: str) -> str:
        try:
            activity_memory_hooks = ShortTermMemoryHook(
                memory_client=client,
                memory_id=memory_id,
                actor_id=activity_actor_id,
                session_id=session_id
            )
            activity_agent = Agent(hooks=[activity_memory_hooks], model=MODEL_ID, system_prompt=ACTIVITY_PLANNER_PROMPT)
            response = activity_agent(query)
            return str(response)
        except Exception as e:
            return f"Error in activity planner assistant: {str(e)}"

    # Coordinator agent
    TRAVEL_AGENT_SYSTEM_PROMPT = (
        "You are a comprehensive travel planning assistant that coordinates between specialized tools:\n"
        "- For flight-related queries (bookings, schedules, airlines, routes) → Use the flight_booking_assistant tool\n"
        "- For hotel-related queries (accommodations, amenities, reservations) → Use the hotel_booking_assistant tool\n"
        "- For activities, tours, or local experiences → Use the activity_planner_assistant tool\n"
        "- For complete travel packages → Use all tools as needed to provide comprehensive information\n"
        "- For general travel advice or simple travel questions → Answer directly\n"
        "Each agent will have its own memory in case the user asks about historic data.\n"
        "When handling complex travel requests, coordinate information from all tools to create a cohesive travel plan.\n"
        "Provide clear organization when presenting information from multiple sources.\n"
        "Ask max two questions per turn. Keep the messages short, don't overwhelm the customer."
    )
    travel_agent = Agent(
        system_prompt=TRAVEL_AGENT_SYSTEM_PROMPT,
        model=MODEL_ID,
        tools=[flight_booking_assistant, hotel_booking_assistant, activity_planner_assistant]
    )

    # Example interaction
    print("=== Travel Planning Multi-Agent System ===")
    print("User: I want to plan a trip to Paris from July 10 to July 20, including flights, hotel, and activities.")
    response = travel_agent("I want to plan a trip to Paris from July 10 to July 20, including flights, hotel, and activities.")
    print(f"Agent: {response}")
    print("User: Suggest some local experiences and tours for those dates.")
    response = travel_agent("Suggest some local experiences and tours for those dates.")
    print(f"Agent: {response}")
    print("User: Book a mid-range hotel near the Eiffel Tower.")
    response = travel_agent("Book a mid-range hotel near the Eiffel Tower.")
    print(f"Agent: {response}")
    print("User: Find a direct flight from New York to Paris.")
    response = travel_agent("Find a direct flight from New York to Paris.")
    print(f"Agent: {response}")

if __name__ == "__main__":
    main()
