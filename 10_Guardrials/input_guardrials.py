
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, RunContextWrapper, input_guardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
from pydantic import BaseModel
from dataclasses import dataclass
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Gemini model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

class MathGuardrail(BaseModel):
    is_math: bool
    reasoning: str

guardrail_agent = Agent(
    name="Math Guardrail",
    instructions="""
Check if the user is asking to solve a math problem or math homework.
Respond in JSON format like: {"is_math": true, "reasoning": "<explanation>"}.
""",
    output_type=MathGuardrail,
    model=model,
)

@input_guardrail
async def math_input_guardrail(ctx: RunContextWrapper[None], input: str, agent: Agent) -> GuardrailFunctionOutput:
    result = await Runner.run(
        guardrail_agent,
        str(input),
        context=ctx.context,
        run_config=config
    )

    is_math = result.final_output.is_math

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=not is_math,  
    )

agent = Agent(
    name="Math Assistant",
    instructions="You are a Help full assistant.",
    input_guardrails=[math_input_guardrail],
    model=model,
)

async def main():
    try:
        result = await Runner.run(
            agent,
            "What is the capital of France?",
            run_config=config
        )
        print(result.final_output)

    except InputGuardrailTripwireTriggered:
        print("Math guardrail tripped - non-math question blocked.")

if __name__ == "__main__":
    asyncio.run(main())
