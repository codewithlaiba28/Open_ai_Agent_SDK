import asyncio
from agents import Agent, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel
import os 
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = LitellmModel(
    model="gemini-2.0-flash",   
    api_key=GEMINI_API_KEY 
)

agent = Agent(
    name="Assistant",
    model=model,
    model_settings=ModelSettings(include_usage=True)  
)

async def main():
    result = await Runner.run(agent, "Tokyo ka weather kaisa hai?")
    print("Agent Output:", result.final_output)

    usage = result.context_wrapper.usage
    print("\n--- Token Usage ---")
    print("Requests:", usage.requests)
    print("Input Tokens:", usage.input_tokens)
    print("Output Tokens:", usage.output_tokens)
    print("Total Tokens:", usage.total_tokens)

asyncio.run(main())
