from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool, trace, SQLiteSession
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


@function_tool
def get_weather(city:str):
    """Get the current weather for a given city."""
    return f"The current weather in {city} is sunny with a temperature of 25°C."

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    session1 = SQLiteSession("conversation_123")
    session2 = SQLiteSession("conversation_123")

    result1 = await Runner.run(agent,"Hi, What is pakistani national anthem",session=session1, run_config=config)
    result2 = await Runner.run(agent,"What is Capital of Pakistan",session=session2, run_config=config)
    print("_"*20)
    print("Session 1:", result1.final_output)
    print("_"*20)
    print("Session 2:", result2.final_output)
    print("_"*20)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())