import asyncio
from agents import Agent, Runner, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, function_tool, trace, SQLiteSession
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession
from sqlalchemy.ext.asyncio import create_async_engine
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
    # In your application, you would use your existing engine
    engine = create_async_engine("sqlite+aiosqlite:///conversations.db")

    agent = Agent("Assistant")
    session = SQLAlchemySession(
        "user-456",
        engine=engine,
        create_tables=True,  # Auto-create tables for the demo
    )

    result = await Runner.run(agent, "Hello", session=session, run_config=config)
    print(result.final_output)

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())