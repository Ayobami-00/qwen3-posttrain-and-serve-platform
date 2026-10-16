"""Torchrun entry point; rank zero alone publishes outputs and journal events."""

import argparse
import json
import os
import signal
import threading
import time
import traceback
from pathlib import Path

from qps.util import atomic_json, process_alive, process_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    owner = json.loads(args.owner)
    launcher = process_identity(os.getppid())

    def watchdog():
        while True:
            time.sleep(0.5)
            if not process_alive(owner) or not process_alive(launcher):
                os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watchdog, daemon=True).start()
    from qps.distributed import SnapshotContext, barrier, initialize, main_rank

    initialize()
    from qps.stages import Context
    from qps.training import train

    ctx = (
        Context(args.state, args.run, args.attempt)
        if main_rank()
        else SnapshotContext(args.snapshot)
    )
    try:
        output = train(ctx, args.checkpoint)
        if main_rank():
            atomic_json(ctx.work / "training-output.json", output)
        barrier()
    except BaseException as exc:
        ctx.emit("worker_error", phase="training", error=str(exc))
        traceback.print_exc()
        raise
    finally:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
