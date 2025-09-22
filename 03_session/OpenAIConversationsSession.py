from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool, OpenAIConversationsSession
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

session = OpenAIConversationsSession(conversation_id="conv_123")

@function_tool
def get_weather(city:str):
    """Get the current weather for a given city."""
    return f"The current weather in {city} is sunny with a temperature of 25°C."

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    tools=[get_weather],
)

result = Runner.run_sync(agent, "Please give me a weather of karachi", run_config=config, session=session)
print(result.final_output)
print(f"Conversation History: {session.get_conversation_history()}")

## Open ai ki key leni ho gi