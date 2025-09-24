from agents import Agent, function_tool, trace, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner
import os, asyncio
from dotenv import load_dotenv

# Env load
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini external client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(model=model, model_provider=external_client)


# ---------- TOOL BANANA ----------
@function_tool
def add_numbers(a: int, b: int) -> int:
    """Do numbers ka sum return karta hai"""
    return a + b


# ---------- AGENT DIRECT BANANA ----------
math_agent = Agent(
    name="MathAgent",
    instructions="Math related queries solve karta hai",
    tools=[add_numbers],   # Tool attach kiya
    model=model
)


# ---------- MAIN RUNNER ----------
async def main():

    run = await Runner.run(math_agent, input="Hi How are you", run_config=config)
    # 10 aur 20 ka sum nikal do
#     Agent Output: 10 aur 20 ka sum 30 hai.
# --- Token Usage ---
# Requests: 2        
# Input Tokens: 67   
# Output Tokens: 21  
# Total Tokens: 88 
    print("Agent Output:", run.final_output)

    # Token usage check karna
    usage = run.context_wrapper.usage
    print("\n--- Token Usage ---")
    print("Requests:", usage.requests)
    print("Input Tokens:", usage.input_tokens)
    print("Output Tokens:", usage.output_tokens)
    print("Total Tokens:", usage.total_tokens)


# Run
asyncio.run(main())
