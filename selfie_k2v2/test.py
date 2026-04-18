import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner

async def main():
    client = AsyncDedalus(api_key="")
    runner = DedalusRunner(client)

    response = await runner.run(
        input="Hello, what can you do?",
        model="anthropic/claude-opus-4-5",
    )
    print(response.final_output)

asyncio.run(main())