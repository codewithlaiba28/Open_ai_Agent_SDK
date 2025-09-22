import asyncio
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool, run_demo_loop
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# External client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model settings
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Function tool
@function_tool
def get_weather(city: str):
    """Get the current weather for a given city."""
    return f"The current weather in {city} is sunny with a temperature of 25°C."

# Agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    tools=[get_weather],
    model=model,
)

async def main():
    await run_demo_loop(agent,)



if __name__ == "__main__":
    asyncio.run(main())
