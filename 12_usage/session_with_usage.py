from agents import Agent, ModelSettings, SQLiteSession,OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool
from dataclasses import dataclass
import os
import asyncio
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
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
    )
    session = SQLiteSession("my_conversation")

    first = await Runner.run(agent, "Hi!", session=session, run_config=config)
    first_usage = first.context_wrapper.usage
    print(f"first_usage input_tokens :   {first_usage.input_tokens}")  
    print(f"first_usage noutput_tokens :   {first_usage.output_tokens}")  
    print(f"first_usage total_tokens :   {first_usage.total_tokens}") 
    print(f"first final_output :   {first.final_output}") 
    second = await Runner.run(agent, "Can you elaborate?", session=session, run_config=config)
    second_usage = second.context_wrapper.usage
    print(f"second_usage input_tokens :   {second_usage.input_tokens}")  
    print(f"second_usage output_tokens :   {second_usage.output_tokens}")  
    print(f"second_usage total_tokens :   {second_usage.total_tokens}") 
    print(f"second final_output :   {second.final_output}") 


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())    
