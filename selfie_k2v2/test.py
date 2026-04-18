import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner

async def main():
    client = AsyncDedalus(api_key="dsk-test-b7859b000ec9-6c69b140d4cfac2e12644ba1309bb236")
    runner = DedalusRunner(client)

    response = await runner.run(
        input="Hello, what can you do?",
        model="anthropic/claude-opus-4-5",
    )
    print(response.final_output)

asyncio.run(main())