from agents import Agent, function_tool, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner
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
class UserContext:
    name: str | None = None
    age: int | None = None

@function_tool
def get_weather(city:str):
    """Get the current weather for a given city."""
    return f"The current weather in {city} is sunny with a temperature of 25°C."

agent = Agent[UserContext](
    name="Assistant", 
    instructions=(
        "You are a helpful assistant. "
        "If the user asks personal questions, always use their name and age from context. "
        "If context is missing, politely ask them to provide it first."
    ),
    tools=[get_weather],
)

# result = Runner.run_sync(agent,"Please give me a weather of karachi", run_config=config)
# #  "Hello, how are you."   answer   I am doing well, thank you for asking. How can I help you today?
# print(result.final_output)


query = input("Enter your query: ")

user_context = UserContext()

if "name" in query.lower() or "age" in query.lower():
    user_context = UserContext(name="Laiba", age=18)

result = Runner.run_sync(agent, query, run_config=config, context=user_context)
print(result.final_output)