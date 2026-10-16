"""One-model reference server with bounded admission, streaming and cancellation."""

import argparse
import asyncio
import concurrent.futures
import contextlib
import json
import queue
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qps.artifacts import verify_artifact
from qps.data import prompt_ids
from qps.specs import RunSpec
from qps.store import Store


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant", "system"]
    content: str


class Completion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: Literal[0] = 0
    stream: bool = False
    n: Literal[1] = 1


class Chat(Completion):
    prompt: None = None
    messages: list[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def last_user(self):
        if self.messages[-1].role != "user":
            raise ValueError("The final message must be a user turn")
        return self


def create_app(store: Store, artifact_id: str, name: str, *, max_context_tokens: int | None = None):
    from transformers import TextIteratorStreamer

    from qps.ml import generate, load_model, tokenizer_from

    active = threading.BoundedSemaphore(1)
    cancel_events: set[threading.Event] = set()
    backend = {}
    loaded_at = int(time.time())

    @asynccontextmanager
    async def lifespan(app):
        directory, manifest = verify_artifact(store, artifact_id)
        if manifest["kind"] != "merged_model":
            raise ValueError("Reference deployments require a merged model artifact")
        run = store.run(manifest["run_id"])
        spec = RunSpec.model_validate_json(run["spec"]).spec
        tokenizer = tokenizer_from(directory)
        model = load_model(directory, spec)
        probe = tokenizer.encode("probe", add_special_tokens=False)
        generate(model, tokenizer, probe, 1)
        backend.update(model=model, tokenizer=tokenizer)
        yield
        for event in list(cancel_events):
            event.set()

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    def health():
        return {"status": "alive"}

    @app.get("/readyz")
    def ready():
        if not backend:
            raise HTTPException(503, "Model is loading")
        record = store.artifact(artifact_id)
        return {
            "status": "ready",
            "artifact_id": artifact_id,
            "model": name,
            "manifest_sha": record["manifest_sha"],
            "backend": "transformers",
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": loaded_at,
                    "owned_by": "qps",
                    "artifact_id": artifact_id,
                }
            ],
        }

    class BoundedStreamer(TextIteratorStreamer):
        def __init__(self, tokenizer, cancel):
            super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
            self.text_queue = queue.Queue(maxsize=16)
            self.cancel = cancel

        def on_finalized_text(self, text, stream_end=False):
            for item in [text, self.stop_signal] if stream_end else [text]:
                while not self.cancel.is_set():
                    try:
                        self.text_queue.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue

    async def respond(body, request: Request, chat: bool):
        if body.model != name:
            raise HTTPException(404, "Unknown model")
        tokenizer = backend["tokenizer"]
        if chat:
            ids = prompt_ids(tokenizer, [message.model_dump() for message in body.messages])
        else:
            ids = tokenizer.encode(body.prompt, add_special_tokens=False)
        context_limit = min(
            backend["model"].config.max_position_embeddings,
            max_context_tokens or backend["model"].config.max_position_embeddings,
        )
        if not ids or len(ids) + body.max_tokens > context_limit:
            raise HTTPException(422, "Prompt plus requested output exceeds the model context limit")
        if not active.acquire(blocking=False):
            raise HTTPException(429, "One generation is already active; retry when it completes")
        cancel = threading.Event()
        cancel_events.add(cancel)
        streamer = BoundedStreamer(tokenizer, cancel) if body.stream else None
        future = concurrent.futures.Future()

        def produce():
            try:
                future.set_result(
                    generate(
                        backend["model"],
                        tokenizer,
                        ids,
                        body.max_tokens,
                        cancel=cancel,
                        streamer=streamer,
                    )
                )
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                cancel_events.discard(cancel)
                active.release()

        threading.Thread(target=produce, daemon=True).start()
        wrapped = asyncio.wrap_future(future)
        # Retrieve an exception even if a disconnected client never awaits the result.
        wrapped.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        identifier = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())

        async def watch_disconnect():
            while not future.done():
                if await request.is_disconnected():
                    cancel.set()
                    return
                await asyncio.sleep(0.05)

        if not body.stream:
            watcher = asyncio.create_task(watch_disconnect())
            try:
                result = await asyncio.shield(wrapped)
                choice = {"index": 0, "finish_reason": result["finish_reason"]}
                choice.update(
                    {"message": {"role": "assistant", "content": result["text"]}}
                    if chat
                    else {"text": result["text"], "logprobs": None}
                )
                return JSONResponse(
                    {
                        "id": identifier,
                        "object": "chat.completion" if chat else "text_completion",
                        "created": created,
                        "model": name,
                        "choices": [choice],
                        "usage": {
                            "prompt_tokens": result["prompt_tokens"],
                            "completion_tokens": result["completion_tokens"],
                            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                        },
                    }
                )
            finally:
                cancel.set()
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher

        def chunk(text="", reason=None, role=False):
            choice = {"index": 0, "finish_reason": reason}
            choice.update(
                {"delta": {"role": "assistant"} if role else {"content": text}}
                if chat
                else {"text": text, "logprobs": None}
            )
            value = {
                "id": identifier,
                "object": "chat.completion.chunk" if chat else "text_completion",
                "created": created,
                "model": name,
                "choices": [choice],
            }
            return "data: " + json.dumps(value) + "\n\n"

        async def stream():
            try:
                if chat:
                    yield chunk(role=True)
                while True:
                    try:
                        item = await asyncio.to_thread(streamer.text_queue.get, True, 0.1)
                    except queue.Empty:
                        if future.done():
                            break
                        continue
                    if item is streamer.stop_signal:
                        break
                    if item:
                        yield chunk(item)
                result = await asyncio.shield(wrapped)
                yield chunk(reason=result["finish_reason"])
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield (
                    "data: "
                    + json.dumps({"error": {"message": str(exc), "type": "generation_error"}})
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
            finally:
                cancel.set()

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completion(body: Completion, request: Request):
        return await respond(body, request, False)

    @app.post("/v1/chat/completions")
    async def chat_completion(body: Chat, request: Request):
        return await respond(body, request, True)

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path)
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--manifest-sha")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fd", type=int)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-context-tokens", type=int)
    args = parser.parse_args()
    if args.artifact_directory:
        from qps.runtime import ArtifactRuntime

        if not args.spec or not args.manifest_sha:
            parser.error("Standalone serving requires --spec and --manifest-sha")
        store = ArtifactRuntime(
            args.artifact_directory,
            args.manifest_sha,
            RunSpec.model_validate_json(args.spec.read_text()),
        )
    else:
        if not args.state:
            parser.error("Provide --state or --artifact-directory")
        store = Store(args.state)
    app = create_app(store, args.artifact, args.name, max_context_tokens=args.max_context_tokens)
    config = uvicorn.Config(app, log_level="info", timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)
    if args.fd is not None:
        server.run(sockets=[socket.socket(fileno=args.fd)])
    else:
        sock = socket.socket()
        sock.bind(("127.0.0.1", args.port))
        server.run(sockets=[sock])


if __name__ == "__main__":
    main()
