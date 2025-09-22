import asyncio
from agents import Agent, Runner, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from agents.memory.session import SessionABC
from agents.items import TResponseInputItem
from typing import List
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


# Custom notebook (memory system)
class MyCustomSession(SessionABC):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory: List[TResponseInputItem] = []  # notebook (list)

    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        if limit:
            return self.memory[-limit:]  # last N items
        return self.memory

    async def add_items(self, items: List[TResponseInputItem]) -> None:
        self.memory.extend(items)  # new conversation likh lena

    async def pop_item(self) -> TResponseInputItem | None:
        if self.memory:
            return self.memory.pop()  # last item delete
        return None

    async def clear_session(self) -> None:
        self.memory.clear()  # puri notebook saaf


# Real-life usage
async def main():
    agent = Agent("Shopkeeper")
    session = MyCustomSession("customer-123")

    # Pehli baat (customer: Hello)
    result1 = await Runner.run(agent, "Hello", session=session, run_config=config)
    print("Agent:", result1.final_output)

    # Dusri baat (customer: What did I say before?)
    result2 = await Runner.run(agent, "What did I say before?", session=session, run_config=config)
    print("Agent:", result2.final_output)


if __name__ == "__main__":
    asyncio.run(main())
