from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool
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
    tracing_disabled=True,
    
)

agent = Agent(
    name="Agent",
    instructions="You are a helpful assistant.",
    model_settings=ModelSettings(temperature= 0.7,          
    top_p =0.8,              
    top_k =40,                 
    max_output_tokens= 200,    
    stop_sequences =["END"],
    # tool_choice="auto"
    # tool_choice="required"
    # tool_choice="none"
    tool_choice="my_first_tool",   
    parallel_tool_calls=False ,
    stop_on_first_tool=True ,
    reset_tools=True 
    )
)
import asyncio

async def main():
    result = await Runner.run(agent, "Hello, how are you? please give me 3 colour name", run_config=config)
    print(result.final_output)

asyncio.run(main())

