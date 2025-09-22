from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool, trace, SQLiteSession
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

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
    agent = Agent("Assistant")
    session = SQLAlchemySession.from_url(
    "user-123",
    url="sqlite+aiosqlite:///my_memory.db",
    create_tables=True,
)

    result = await Runner.run(agent, "Hello", session=session, run_config=config)

    print("Result:", result.final_output)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())