# Python test boundaries

Run commands from the repository root. The committed lock selects dependencies;
`--no-sync` keeps the extras already installed in the environment.

## Fast input, artifact and journal checks

These groups use small dictionaries, deterministic tokenizer doubles and temporary
files/SQLite databases. They do not load models, download datasets, contact cloud
APIs or initialize CUDA. Core dependencies and the development group are enough:

```bash
uv sync --project python --python 3.12 --frozen --group dev
uv run --project python --no-sync pytest python/tests/test_data_*.py python/tests/test_artifact_*.py python/tests/test_store_*.py -q
```

| Tests | Behavior protected |
| --- | --- |
| [Numeric answers](test_data_numeric.py) | Exact Fraction comparison, complete final-answer suffix and bounded parsing |
| [Normalization](test_data_normalization.py) | Unicode/whitespace grouping, original prompt preservation and annotation removal |
| [Supervision](test_data_supervision.py) | Non-thinking template flags, response labels, EOS and exact sequence budget |
| [Preparation](test_data_preparation.py) | Stable split membership, deterministic duplicate selection and rejection evidence |
| [Manifest envelopes](test_artifact_manifest.py) | Safe canonical paths, identities, sizes and SHA-256 values |
| [Artifact files](test_artifact_files.py) | Exact inventory, same-size corruption, symbolic files and empty payload files |
| [Publication](test_artifact_publication.py) | Interrupted copies, unregistered corrupt outputs and repeat reconciliation |
| [Standalone runtime](test_artifact_runtime.py) | Verified relocation without the old registry and rejection of unrelated identities |
| [Journal durability](test_store_durability.py) | Committed events after projection failure and recursive non-finite rejection |
| [Journal queries](test_store_queries.py) | Exclusive run-scoped cursors, attempt fencing and committed phase outputs |

`character_tokenizer` in [conftest.py](conftest.py) makes token boundaries visible:
two prompt tokens, one token per response character and a distinct end token.
It checks preprocessing, not Qwen tokenization or model quality. `local_run`
creates a recipe/journal without executing training. The `successful` fixture
instead runs the real CPU smoke and is loaded only when a test requests it.

The standalone-runtime tests use arbitrary byte payloads to verify the bundle
boundary. They do not prove that those bytes form a loadable Safetensors model.

## Complete CPU suite

```bash
uv sync --project python --python 3.12 --frozen --extra train --extra serve --extra hosted --group dev
uv run --project python --no-sync ruff check python/src python/tests
uv run --project python --no-sync ruff format --check python/src python/tests
uv run --project python --no-sync pytest python/tests -m 'not gpu' -q
```

Some CPU tests perform real tiny-model training and loopback HTTP/S3-emulator
workflows. The `integration` marker identifies the real CPU workflow module;
selecting `not gpu` does not mean every test is a pure unit test.

PostgreSQL tests require `QPS_TEST_POSTGRES_URL`; absent that setting, their skips
do not establish migration or database-concurrency acceptance. Browser/Kind and
live-provider gates are documented in the [end-to-end guide](../../tests/e2e/README.md).
The `gpu` marker requires separate explicit CUDA opt-in and an already provisioned
compatible host. None of these commands establishes real FSDP2/NCCL acceptance.

Test fixtures belong under pytest temporary directories or ignored `.qps/`.
Keep signed object URLs, owner credentials, model weights and live service tokens
out of assertions, captured output and committed fixture data. Release evidence
must distinguish the test's source/image identity and which gate actually ran.
