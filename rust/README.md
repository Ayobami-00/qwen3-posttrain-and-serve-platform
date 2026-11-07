# qps-inference

The Rust crate currently implements startup-document parsing and value
validation. It does not load weights, initialize CUDA or start an HTTP server.
The platform continues to use the Python reference server.

## Run the approved example

From the repository root:

```bash
cargo run --manifest-path rust/Cargo.toml --locked -- --runtime contracts/fixtures/inference/runtime.json
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo test --manifest-path rust/Cargo.toml --locked
```

After dependencies are cached, append `--offline` to the Cargo options to avoid
registry access. The example contains synthetic resource IDs, a placeholder hash
and a model path with no corresponding bundle. Passing this example establishes
document validity only.

## Current execution paths

```text
CLI arguments → read_runtime → Runtime → validate_runtime → Debug output

Embedded backend JSON → load_backend_capabilities
                     → BackendCapabilities → validate_backend_capabilities
```

The second path is available through the public library and integration tests;
`main` does not yet call it or compare the two documents.

| Source | Responsibility |
| --- | --- |
| [main.rs](src/main.rs) | Required `--runtime` argument, diagnostics and process exit status |
| [lib.rs](src/lib.rs) | Re-export the public types and loading/validation functions |
| [contracts.rs](src/contracts.rs) | Owned Serde structures and contract enums |
| [validation.rs](src/validation.rs) | File loading, embedded descriptor loading and value checks |
| [tests](tests) | Public API and executable regression tests |

Serde checks required fields, names, types and enum values. Explicit validators
then check versions, identifiers, numeric limits and nonempty/unique capability
lists. These checks are written in Rust; the executable does not run a JSON
Schema validator. The agreed [schemas and fixtures](../contracts/README.md) remain
the cross-language specification.

`ContractLoadError::Read` retains an owned path and I/O cause. `Parse` retains a
Serde cause and an optional path: `Some(path)` for a runtime file, `None` for the
embedded backend descriptor. Invalid UTF-8 file contents are an I/O error from
`read_to_string`. Command-line paths use `args_os` and may be non-Unicode on Unix.

The CLI returns 0 on success, 1 on loading/value failure and 2 on argument misuse.
Diagnostics go to stderr; successful runtime inspection goes to stdout.

## Compatibility still to implement

Document validity does not establish backend support. The runtime schema permits
CPU intent and TP=2, while the initial backend target advertises CUDA/BF16/TP=1.
The next check must match the requested execution tuple and model profile to the
descriptor and enforce its limits.

Later checks will load the model profile, verify original manifest bytes and file
checksums, and inspect model configuration/tensor metadata. Tokenization, GPU
execution, generation and HTTP serving follow those checks. Descriptor fields
such as `streaming: true` describe the target interface, not completed engine
capabilities. Production Rust implementation is being developed by the project
owner; the regression tests exercise the interfaces already exposed.
