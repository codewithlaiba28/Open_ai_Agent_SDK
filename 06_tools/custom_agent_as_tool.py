import asyncio
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool, FunctionTool
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
async def run_my_agent() -> str:
    """A tool that runs the agent with custom configs"""

    agent = Agent(name="My agent", instructions="You are helpful assistant")

    result = await Runner.run(
        agent,
        input="Hello, how are you?",
        run_config=config
    )

    print(result.final_output)
    return str(result.final_output)
agent = Agent(
    name="Main Agent",
    instructions="You can call tools to do tasks.",
    tools=[run_my_agent],   # 👈 tool yahan pass karna hai
)

if __name__ == "__main__":
    result = asyncio.run(
        Runner.run(agent, "Please run the custom agent", run_config=config)
    )
    print(result.final_output)
