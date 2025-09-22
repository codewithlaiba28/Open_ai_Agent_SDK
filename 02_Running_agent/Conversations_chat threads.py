from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool, trace
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

    thread_id = "thread_123"  
    with trace(workflow_name="Conversation", group_id=thread_id):

        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?", run_config=config)
        print(result.final_output)

        new_input = result.to_input_list() + [{"role": "user", "content": "What state is it in?"}]
        result = await Runner.run(agent, new_input, run_config=config)
        print(result.final_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())