from agents import Agent, Runner, handoff
from agents.extensions import handoff_filters
from agents import ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import os
from dotenv import load_dotenv

# Load Gemini API Key
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

# Sub-agents
billing_agent = Agent(
    name="Billing agent",
    instructions="Handle all billing related queries such as invoices, payments, and account charges."
)

refund_agent = Agent(
    name="Refund agent",
    instructions="Handle refund related queries such as money back requests, transaction reversals, or failed payments."
)

billing_handoff = handoff(
    agent=billing_agent,
    input_filter=handoff_filters.remove_all_tools,  # remove tools for clean routing
)

refund_handoff = handoff(
    agent=refund_agent,
    input_filter=handoff_filters.remove_all_tools,
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Identify the type of query and hand it off to the correct department (billing or refund).",
    handoffs=[billing_handoff, refund_handoff]
)

# Run test
result = Runner.run_sync(triage_agent, "Please return my order", run_config=config)
print(result.final_output)
