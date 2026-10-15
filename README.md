# Qwen3 Post-Training and Serving Platform

An open-source platform for post-training, evaluating and serving Qwen3 models, with a planned local GPU workflow, Kubernetes-managed Runpod compute, experiment tracking and a Rust inference runtime.

**Status: architecture and implementation planning.** Training, serving, orchestration and dashboard functionality are not implemented yet.

## Design and implementation plan

The first milestone is one complete workflow: prepare real data, train a LoRA adapter, export and evaluate the model, register its provenance and serve the exact checkpoint over HTTP. A Transformers reference backend establishes correctness in v0.1; actual Rust model execution follows in v0.2.

The hosted reference design uses AWS for persistent platform services and Runpod for GPU compute. Local use remains independent of either account.

## License and project status

The code is licensed under [Apache 2.0](LICENSE). Downloaded models, datasets and dependencies retain their own licenses and attribution requirements.

This is an independent community project and is not affiliated with or endorsed by the Qwen team.
