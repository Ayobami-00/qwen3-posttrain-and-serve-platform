"""Isolated phase worker with orphan protection if its coordinator disappears."""

import argparse
import json
import os
import signal
import threading
import time
import traceback
from pathlib import Path

from qps.util import atomic_json, process_alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    owner = json.loads(args.owner)

    def watchdog():
        while True:
            time.sleep(0.5)
            if not process_alive(owner):
                # The worker owns its isolated process group (start_new_session=True).
                os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watchdog, daemon=True).start()
    from qps import stages

    ctx = stages.Context(args.state, args.run, args.attempt)
    try:
        if args.phase == "training":
            from qps.training import train

            output = train(ctx, args.checkpoint)
        else:
            output = getattr(stages, args.phase)(ctx)
        atomic_json(ctx.work / f"{args.phase}-output.json", output)
    except BaseException as exc:
        ctx.emit("worker_error", phase=args.phase, error=str(exc))
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
