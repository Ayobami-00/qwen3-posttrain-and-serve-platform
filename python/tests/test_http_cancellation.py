import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from qps.hosted.http import until_disconnect


def test_normal_response_does_not_wait_for_disconnect_watcher():
    async def scenario():
        async def receive():
            await asyncio.Event().wait()

        request = Request({"type": "http"}, receive=receive)
        for _ in range(20):
            assert (
                await asyncio.wait_for(
                    until_disconnect(request, asyncio.sleep(0.001, result=42)), 0.2
                )
                == 42
            )

    asyncio.run(scenario())


def test_disconnect_cancels_upstream_before_response_headers():
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class Client:
            closed = False

            async def is_disconnected(self):
                return self.closed

        async def slow_headers():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        client = Client()
        task = asyncio.create_task(until_disconnect(client, slow_headers()))
        await started.wait()
        client.closed = True
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(task, 0.3)
        assert exc.value.status_code == 499
        assert cancelled.is_set()

    asyncio.run(scenario())
