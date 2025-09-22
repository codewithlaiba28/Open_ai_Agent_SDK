# import os
# import asyncio
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from agents import (
#     Agent, 
#     function_tool,
#     Runner,
#     TResponseInputItem,
#     RunConfig, 
#     OpenAIChatCompletionsModel,
#     AsyncOpenAI,
#     input_guardrail, 
#     output_guardrail, 
#     RunContextWrapper, 
#     InputGuardrailTripwireTriggered,
#     OutputGuardrailTripwireTriggered,
#     GuardrailFunctionOutput
# )
# from pydantic import BaseModel

# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# external_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# class MessageOutput(BaseModel): 
#     response: str

# class CheckMathResponse(BaseModel):
#     is_math: bool
#     reasoning: str

# class MathGuardrail(BaseModel):
#     is_math: bool
#     reasoning: str

# input_guardrail_agent = Agent(
#     name="Math Guardrail",
#     instructions="""
#         Detect if the user is asking a math problem.

#         - If input mentions solving, calculating, or has numbers/equations with +, -, *, /, ^, = → {"is_math": true, "reasoning": "Math detected"}.
#         - If unsure, default to {"is_math": true, "reasoning": "Unclear but possibly math"}.
#         - Only mark {"is_math": false} when it is very clear that the question is NOT math.
#         Always respond ONLY in JSON.
#     """,
#     output_type=MathGuardrail,
#     model=model,
# )

# Output_guardrail_agent = Agent(
#     name="Guardrail check",
#     instructions="""
# Check if the assistant's output contains math equations, calculations, or solutions (like x=4, 2+2=4, etc).
# If yes, respond in JSON: {"is_math": true, "reasoning": "Output contains math"}.
# If no, respond: {"is_math": false, "reasoning": "Output has no math"}.
# """,
#     output_type=CheckMathResponse,
#     model=model,
# )

# @input_guardrail
# async def math_input_guardrail(ctx: RunContextWrapper[None], input: str, agent: Agent) -> GuardrailFunctionOutput:
#     result = await Runner.run(
#         input_guardrail_agent,
#         str(input),
#         context=ctx.context,
#         run_config=config
#     )

#     is_math = result.final_output.is_math

#     return GuardrailFunctionOutput(
#         output_info=result.final_output, 
#         tripwire_triggered=not is_math,  
#     )

# @output_guardrail
# async def math_ouput_guardrials(  
#     ctx: RunContextWrapper, agent: Agent, output: MessageOutput
# ) -> GuardrailFunctionOutput:
#     result = await Runner.run(output_guardrail, output.response, context=ctx.context)

#     return GuardrailFunctionOutput(
#         output_info=result.final_output,
#         tripwire_triggered=result.final_output.is_math,
#     )

# agent = Agent(
#     name="Math Assistant",
#     instructions="You are a Help full assistant.",
#     input_guardrails=[math_input_guardrail],
#     output_guardrails=[math_ouput_guardrials],
#     model=model,
# )

# async def main():
#     try:
#         result = await Runner.run(
#             agent,
#             "solve for x: 2x + 3 = 11?",
#             run_config=config
#         )
#         print(result.final_output)

#     except InputGuardrailTripwireTriggered:
#         print("Math input guardrail tripped - non-math question blocked.")

#     except OutputGuardrailTripwireTriggered:
#         print("Math Output guardrail tripped - math answer blocked.")    

# if __name__ == "__main__":
#     asyncio.run(main())

import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Any

from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    input_guardrail,
    output_guardrail,
    RunContextWrapper,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    GuardrailFunctionOutput,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)


# ---------- pydantic output types ----------
class MessageOutput(BaseModel):
    response: str


class CheckMathResponse(BaseModel):
    is_math: bool
    reasoning: str


class MathGuardrail(BaseModel):
    is_math: bool
    reasoning: str


# ---------- guardrail agents ----------
input_guardrail_agent = Agent(
    name="Math Input Guardrail",
    instructions="""
        Detect if the user input is a math problem.
        - If input clearly contains a math question (equations, requests to solve, numeric expressions),
          respond JSON: {"is_math": true, "reasoning": "..."}.
        - If unsure, default to {"is_math": false, "reasoning": "Unclear / not math"}.
        ALWAYS respond ONLY in JSON.
    """,
    output_type=MathGuardrail,
    model=model,
)

output_guardrail_agent = Agent(
    name="Math Output Guardrail",
    instructions="""
        Check whether the assistant's output contains explicit math (equations, calculations, numeric solutions).
        - If yes: {"is_math": true, "reasoning": "..."}.
        - If no: {"is_math": false, "reasoning": "..."}.
        ALWAYS respond ONLY in JSON.
    """,
    output_type=CheckMathResponse,
    model=model,
)


# ---------- guardrail functions ----------
@input_guardrail
async def math_input_guardrail(
    ctx: RunContextWrapper[None], user_input: str, agent: Agent
) -> GuardrailFunctionOutput:
    """Allow only math inputs. Trip (block) when input is NOT math."""
    result = await Runner.run(
        input_guardrail_agent,
        str(user_input),
        context=ctx.context,
        run_config=config,
    )

    is_math = bool(result.final_output.is_math)

    # Accept only math inputs → trip if NOT math
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not is_math),
    )


@output_guardrail
async def math_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: Any
) -> GuardrailFunctionOutput:
    """
    Block assistant outputs that contain math.
    This function is robust to the runtime passing a str, dict, or MessageOutput-like object.
    """
    # --- Extract plain text from whatever 'output' is ---
    if isinstance(output, MessageOutput):
        text = output.response
    elif isinstance(output, dict):
        # common case: the runtime may pass a dict with 'response' or 'text'
        text = output.get("response") or output.get("text") or str(output)
    else:
        # fallback: maybe it's a plain string or some other object
        text = str(output)

    # Run the output-check guardrail agent on the extracted text
    result = await Runner.run(
        output_guardrail_agent,
        text,
        context=ctx.context,
        run_config=config,
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=bool(result.final_output.is_math),
    )


# ---------- main agent ----------
agent = Agent(
    name="Math-only-in, No-math-out Assistant",
    instructions=(
        "You only receive math questions. When replying, DO NOT include numeric calculations, "
        "equations, or explicit solved values. Instead give a conceptual, non-numeric explanation."
    ),
    input_guardrails=[math_input_guardrail],
    output_guardrails=[math_output_guardrail],
    model=model,
)


# ---------- run ----------
async def main():
    try:
        result = await Runner.run(
            agent,
            "Solve for x: 2x + 3 = 11",  # math input -> should be allowed
            run_config=config,
        )
        print("Agent final_output:", result.final_output)

    except InputGuardrailTripwireTriggered:
        print("Blocked: input is not math (input guardrail tripped).")

    except OutputGuardrailTripwireTriggered:
        print("Blocked: assistant output contained math (output guardrail tripped).")


if __name__ == "__main__":
    asyncio.run(main())
