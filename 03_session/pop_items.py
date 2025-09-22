from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool, SQLiteSession
import asyncio
from dataclasses import dataclass
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

async def main():
    agent = Agent(name="Assistant")
    session = SQLiteSession("correction_example")

    result = await Runner.run(agent,"What's 2 + 2?", run_config=config, session=session)

    print(f"Agent: {result.final_output}")
    user_item = await session.pop_item()  

    assistant_item = await session.pop_item()  
        
    result = await Runner.run(agent,   "What's 2 + 3?",run_config=config, session=session)
    print(f"Agent: {result.final_output}")

    print(f"Popped items from session: {user_item.get('text')}, {assistant_item.get('text')}") 
if __name__ == "__main__":
    asyncio.run(main())