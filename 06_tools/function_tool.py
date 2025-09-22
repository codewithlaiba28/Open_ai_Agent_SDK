from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool, RunContextWrapper, FunctionTool
import asyncio
import json
from typing_extensions import TypedDict, Any
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

class Location(TypedDict):
    city: str
    state: str
    country: str

@function_tool
async def get_weather(location: Location) -> str:
    """Get the current weather for a given city."""
    return f"The current weather in {location} is sunny with a temperature of 25°C."


@function_tool(name_override="fetch_data")  
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    # In real life, we'd read the file from the file system
    return "<06_tools/data.txt> file contents"


agent = Agent(
    name="Assistant",
    tools=[get_weather, read_file],  
)

Result = Runner.run_sync(agent, "What's the weather in New York?", run_config=config)

print(Result.final_output)

for tool in agent.tools:
    if isinstance(tool, FunctionTool):
        print(tool.name)
        print(tool.description)
        print(json.dumps(tool.params_json_schema, indent=2))
        print()