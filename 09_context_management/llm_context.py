import asyncio
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, RunContextWrapper, function_tool
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
@dataclass
class UserInfo:  
    name: str

@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:  
    """Fetch the age of the user. Call this function to get user's age information."""
    return f"The user {wrapper.context.name} is 18 years old"

async def main():
    agent = Agent(
        name="Assistant",
        instructions="""
        You are a helpful assistant.
        The user will tell you their name in the prompt.
        Your job is to read the name directly from the prompt
        and always answer with their age as 18 years old.
        """,
        tools=[fetch_user_age],
    )

    result = await Runner.run(
        agent,"My name is liaba and whats my age",run_config=config
    )

    print(result.final_output)  

if __name__ == "__main__":
    asyncio.run(main())
