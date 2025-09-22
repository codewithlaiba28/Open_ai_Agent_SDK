from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool, RunContextWrapper
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
    name: str 
    age: int 

def dynamic_instructions(
    context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."

@function_tool
def get_weather(city: str):
    """Get the current weather for a given city."""
    return f"The current weather in {city} is sunny with a temperature of 25°C."

agent = Agent[UserContext](
    name="Assistant",
    instructions=dynamic_instructions,
    tools=[get_weather],
)

# 👇 User context banana zaroori hai
user_context = UserContext(name="Laiba", age=25)

result = Runner.run_sync(
    agent,
    "My name is Laiba. Please give me the weather of Karachi",
    run_config=config,
    context=user_context  # 👉 ab pass ho raha hai
)

print(result.final_output)
