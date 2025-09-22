from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool
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

@function_tool
def book_ticket(city: str, date: str):
    print(f"Booking ticket ")
    return f"Ticket booked for {city} on {date}"

@function_tool
def refund_ticket(ticket_id: str):
    print(f"Refund ")

    return f"Refund processed for ticket {ticket_id}"

booking_agent = Agent(
    name="Booking agent",
    instructions="Handle all booking related questions and requests.",
    tools=[book_ticket],
    model=model
)

refund_agent = Agent(
    name="Refund agent",
    instructions="Handle all refund related questions and requests.",
    tools=[refund_ticket],
    model=model
)

customer_facing_agent = Agent(
    name="Customer support agent",
    instructions="""Handle all direct user communication. Call booking/refund experts as needed.and call the tools to complete the tasks. 
    
    ## Available tools:
    - booking_expert: For booking tickets.
    - refund_expert: For processing refunds
    """,
    model=model,
    tools=[
        booking_agent.as_tool(
            tool_name="booking_expert",
            tool_description="Handles booking questions and requests.",
        ),
        refund_agent.as_tool(
            tool_name="refund_expert",
            tool_description="Handles refund questions and requests.",
        )
    ],
)
async def main():
    result = await Runner.run(
        customer_facing_agent,
        "I want to book a ticket to Paris on 2024-12-25 and also refund my ticket with ID 12345.",
        run_config=config
    )
    print(result.final_output)



asyncio.run(main())