import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, RunConfig, trace

# Load env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini external client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model settings
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Agent
agent = Agent(
    name="Support Agent",
    instructions="You are a helpful customer support agent. Answer politely.",
    model=model,
)
async def main():
    with trace("Customer service workflow") as tr:
        config = RunConfig(model=model)

        result = await Runner.run(agent, "Mera bill kitna hai?", run_config=config)

        print("=== Agent ka Answer ===")
        print(result.final_output)

        print("\n=== Trace Details ===")
        print("Trace ID:", tr.trace_id)
        print("Workflow Name:", tr.name)
        
        # tr.start(mark_as_current=True)
        # print("Running manual trace...")  ye khod nahi deta balke ye apne under store karta hai
        # tr.finish(reset_current=True)
        # FIX: universal way to see trace data
        exported = tr.export()
        print("\n=== Exported Trace ===")
        print(exported)


# async def main():
#     with trace("Customer service workflow") as tr:
#         config = RunConfig(model=model)

#         result = await Runner.run(agent, "Mera bill kitna hai?", run_config=config)

#         print("=== Agent ka Answer ===")
#         print(result.final_output)

#         print("\n=== Trace Details ===")
#         print("Trace ID:", tr.trace_id)
#         print("Workflow Name:", tr.name)


#         print("Total Items:", (tr.name))
#         print("Total Items:", (tr.finish))
#         print("Total Items:", (tr.start))
#         print("Total Items:", (tr.trace_id))
#         print("Total Items:", (tr.export()))


        


asyncio.run(main())
