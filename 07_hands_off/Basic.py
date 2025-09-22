from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool
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

from agents import Agent, handoff

billing_agent = Agent(
    name="Billing agent",
    instructions="Handle all billing related queries such as invoices, payments, and account charges."
)

refund_agent = Agent(
    name="Refund agent",
    instructions="Handle refund related queries such as money back requests, transaction reversals, or failed payments."
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Identify the type of query and hand it off to the correct department (billing or refund).",
    handoffs=[billing_agent, handoff(refund_agent)
    ]
)


result = Runner.run_sync(triage_agent, "Please returned my order", run_config=config)
print(result.final_output)
