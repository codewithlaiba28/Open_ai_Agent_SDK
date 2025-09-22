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

from agents import Agent, handoff, RunContextWrapper, Runner

def on_handoff(ctx: RunContextWrapper[None]):
    print("⚡ Handoff called!")

billing_agent = Agent(
    name="Billing agent",
    instructions="Handle all billing queries like invoices, payments, or account charges."
)

refund_agent = Agent(
    name="Refund agent",
    instructions="Handle refund queries like money back or transaction reversals."
)

refund_handoff = handoff(
    agent=refund_agent,
    on_handoff=on_handoff, 
    tool_name_override="custom_refund_handoff",
    tool_description_override="Handles refund related issues via custom handoff."
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Identify the type of query and route it to billing or refund.",
    handoffs=[
        handoff(
            billing_agent,
        ),
        refund_handoff  
    ]
)

if __name__ == "__main__":
    result1 = Runner.run_sync(triage_agent, "I need help with my bill" ,run_config=config)
    print("Final output:", result1.final_output)

    # Example refund query
    result2 = Runner.run_sync(triage_agent, "I want a refund for my last payment", run_config=config)
    print("Final output:", result2.final_output)
