import logging
from datetime import datetime
import os
from strands import Agent, tool
from strands.hooks import AgentInitializedEvent, HookProvider, HookRegistry, MessageAddedEvent
from bedrock_agentcore.memory import MemoryClient
from botocore.exceptions import ClientError
from ddgs.exceptions import DDGSException, RatelimitException
from ddgs import DDGS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sample-strands-agent")

# Configuration
REGION = os.getenv('AWS_REGION', 'us-west-2')
ACTOR_ID = "user_123"
SESSION_ID = "personal_session_001"

# Web search tool
def websearch(keywords: str, region: str = "us-en", max_results: int = 5) -> str:
    try:
        results = DDGS().text(keywords, region=region, max_results=max_results)
        return str(results) if results else "No results found."
    except RatelimitException:
        return "Rate limit reached. Please try again later."
    except DDGSException as e:
        return f"Search error: {e}"
    except Exception as e:
        return f"Search error: {str(e)}"

# Memory resource setup
client = MemoryClient(region_name=REGION)
memory_name = "SampleAgentMemory"
memory_id = None
try:
    memory = client.create_memory_and_wait(
        name=memory_name,
        strategies=[],
        description="Short-term memory for sample agent",
        event_expiry_days=7,
    )
    memory_id = memory['id']
    logger.info(f"✅ Created memory: {memory_id}")
except ClientError as e:
    logger.info(f"❌ ERROR: {e}")
    if e.response['Error']['Code'] == 'ValidationException' and "already exists" in str(e):
        memories = client.list_memories()
        memory_id = next((m['id'] for m in memories if m['id'].startswith(memory_name)), None)
        logger.info(f"Memory already exists. Using existing memory ID: {memory_id}")
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

# Memory hook provider
class MemoryHookProvider(HookProvider):
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
                        context_messages.append(f"{role}: {content}")
                context = "\n".join(context_messages)
                event.agent.system_prompt += f"\n\nRecent conversation:\n{context}"
                logger.info(f"✅ Loaded {len(recent_turns)} conversation turns")
        except Exception as e:
            logger.error(f"Memory load error: {e}")
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
            logger.error(f"Memory save error: {e}")
    def register_hooks(self, registry: HookRegistry):
        registry.add_callback(MessageAddedEvent, self.on_message_added)
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)

# Create the agent
def create_sample_agent():
    agent = Agent(
        name="SamplePersonalAgent",
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        system_prompt=f"""You are a helpful personal assistant with web search capabilities.\n\nYou can help with:\n- General questions and information lookup\n- Web searches for current information\n- Personal task management\n\nWhen you need current information, use the websearch function.\nToday's date: {datetime.today().strftime('%Y-%m-%d')}\nBe friendly and professional.""",
        hooks=[MemoryHookProvider(client, memory_id, ACTOR_ID, SESSION_ID)],
        tools=[websearch],
    )
    return agent

if __name__ == "__main__":
    agent = create_sample_agent()
    logger.info("✅ Sample agent created with memory and web search.")
    # Example interaction
    print("=== First Conversation ===")
    print("User: My name is Alex and I'm interested in learning about AI.")
    print("Agent: ", end="")
    agent("My name is Alex and I'm interested in learning about AI.")
    print("User: Can you search for the latest AI trends in 2025?")
    print("Agent: ", end="")
    agent("Can you search for the latest AI trends in 2025?")
