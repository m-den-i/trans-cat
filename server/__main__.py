import asyncio
from server.app import get_service


async def main():
    service = await get_service()
    await service.wait_closed()

asyncio.run(main())