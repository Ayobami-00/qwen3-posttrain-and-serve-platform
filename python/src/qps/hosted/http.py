"""Propagate disconnects while an upstream model is still preparing response headers."""

import asyncio

from fastapi import HTTPException


async def until_disconnect(request, operation):
    stopping = False

    async def disconnected():
        while not stopping and not await request.is_disconnected():
            await asyncio.sleep(0.05)

    pending = asyncio.create_task(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((pending, watcher), return_when=asyncio.FIRST_COMPLETED)
        if pending in done:
            return await pending
        raise HTTPException(499, "Client disconnected")
    finally:
        stopping = True
        for task in (pending, watcher):
            if not task.done():
                task.cancel()
        await asyncio.gather(pending, watcher, return_exceptions=True)
