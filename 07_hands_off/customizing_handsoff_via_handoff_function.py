import asyncio
from pydantic import BaseModel
from agents import Agent, handoff, RunContextWrapper

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
    tracing_disabled=True
)
# Step 1: Input model for escalation
class EscalationData(BaseModel):
    reason: str


# Step 2: Escalation agent callback
async def on_handoff(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"\n🚨 Escalation agent called with reason: {input_data.reason}\n")


# Step 3: Create Escalation agent
escalation_agent = Agent(name="Escalation Agent")


# Step 4: Define handoff tool
handoff_tool = handoff(
    agent=escalation_agent,
    on_handoff=on_handoff,
    input_type=EscalationData,
)


# Step 5: Support agent
support_agent = Agent(
    name="Support Agent",
    tools=[handoff_tool],
)


# Step 6: Live chat simulation
async def chat():
    print("🤖 Support Agent: Hello! How can I help you today?")
    
    while True:
        user_input = input("💬 You: ")

        # Exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Chat ended.")
            break

        # Escalation condition
        if "supervisor" in user_input.lower() or "complaint" in user_input.lower():
            reason = EscalationData(reason=user_input)
            await on_handoff(RunContextWrapper(None), reason)
        else:
            agent = Agent(
                name="Support Agent",
                instructions="Respond to customer queries politely and helpfully.",
            
            )
            Result = await Runner.run(agent, user_input, run_config=config)
            print(f"🤖 Support Agent: {Result.final_output}")


if __name__ == "__main__":
    asyncio.run(chat())
