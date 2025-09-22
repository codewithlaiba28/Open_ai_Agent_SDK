import asyncio
from agents import Agent, Runner, handoff, RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# Billing Agent
billing_agent = Agent(
    name="Billing agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a helpful billing support agent.
    Handle queries related to invoices, payments, and account charges.
    If the query is not related to billing, hand it off to the appropriate agent.""",
)

# General Support Agent
general_agent = Agent(
    name="General Support agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a general support assistant.
    Answer queries about account setup, features, or other non-billing issues.""",
)

# Triage Agent (decides where to send query)
triage_agent = Agent(
    name="Triage agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a triage agent.
    If the query is related to billing (invoices, payments, charges), hand it off to the Billing Agent.
    Otherwise, hand it off to the General Support Agent.""",
)

# Handoff callback (optional)
def on_handoff(ctx: RunContextWrapper[None]):
    print("⚡ Handoff triggered... forwarding to correct agent.")

async def main():
    # Runner execute karega agents ko
    runner = Runner(agents=[triage_agent, billing_agent, general_agent])

    # Example Queries
    queries = [
        "Mera last invoice mujhe nahi mila, check karo.",
        "Mujhe account create karne ka process samjhayein.",
        "Meri payment fail ho gayi hai."
    ]

    for q in queries:
        print(f"\nUser Query: {q}")
        result = await runner.run(triage_agent, q)
        print(f"Agent Reply: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
