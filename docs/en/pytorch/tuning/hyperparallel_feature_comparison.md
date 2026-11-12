# HyperParallel and MindSpeed-LLM feature comparison

Scope: [HyperParallel c108cf9](https://github.com/mindspore-ai/hyper-parallel/tree/c108cf94bed1c931f2a49becaf289398481745a8),
[MindSpeed-LLM b82aedb](https://github.com/YueZhou-oh/MindSpeed-LLM/tree/b82aedb6d79f9f0060e489c8aceffb3b709e568a),
and [MindSpeed Core c047aaa](https://github.com/Ascend/MindSpeed/tree/c047aaad23c8cb97e70198b1f06701562a8141ff).
The latter was inspected separately because MindSpeed-LLM imports its feature patches at runtime. The MindSpeed installation on a training node can differ from this revision.

| Area | HyperParallel | MindSpeed-LLM at the baseline | Result |
|---|---|---|---|
| MoE dispatch | Native MegaMoE with SHMEM push/pull, dynamic capacity, and a fused dispatch–GMM–SwiGLU–GMM–combine path | MCore all-to-all, permutation fusion, FB overlap and expert MC2. FSDP2 eager, fused, and MC2 dispatchers. | MegaMoE itself was missing. This change adds its FSDP2 Qwen3 adapter. |
| MoonEP | No implementation with this name was found in the reviewed source | No implementation found | An exact upstream repository or revision is needed to identify this feature. |
| Expert load balance | Expert bias, auxiliary losses, and monitoring | Expert bias and auxiliary losses in MCore. Core has a hot-expert replication feature outside the fork's registered feature list. | Existing routing features overlap. Hot-expert replication requires separate integration. |
| Muon | AdamW companion optimizer, layout-driven sharded updates, batched Newton–Schulz, coefficient schedules, and transform hooks | MCore Muon and layer-wise distributed Muon with TP modes and coefficient options. FSDP2 has Muon plus AdamW and legacy local-shard updates. | This change adds optional Asym5 over logical matrices to FSDP2. It uses gather, compute, and reshard. HyperParallel's batched relay and owner-compute strategy remain separate. |
| Specialized optimizer | Sinkhorn optimizer and DeepSeek-V4.1 head-wise Muon integration | Router and mHC Sinkhorn operations serve different purposes. MCore Muon has QKV splitting. | The separate Sinkhorn optimizer and DeepSeek-V4.1-specific head-wise adapter remain gaps. |
| Optimizer offload | Swapped optimizer state, including SwapMuon | Core contains swap-optimizer and SwapMuon code. The LLM fork has backend-specific registration and constraints. | Inspect the installed backend and supported combinations before attributing this gap to the whole stack. |
| Distributed tensor and TP | DTensor, declarative TP, custom distributed ops, and layout rules | MCore module-level TP and Ascend communication fusions. FSDP2 uses PyTorch DTensor and FSDPTurbo. | The HyperParallel layout and custom-op registry are distinct abstractions. |
| DP and FSDP | HSDP/FSDP sharding, prefetch, and overlap | MCore distributed optimizer. Separate FSDP2 backend with FSDPTurbo. | Both stacks provide sharding and overlap through different entry points. |
| Pipeline parallelism | GPipe, 1F1B, interleaved 1F1B, PP+FSDP MetaStep, and Mpipe transpose | MCore Megatron schedules, DualPipeV, and MindSpeed scheduling extensions | PP itself is available. The MetaStep and Mpipe integrations are distinct ports. |
| Context parallelism | Async CP and DSA paths | Ulysses, other MCore CP modes, and model-specific DSA implementations | Existing coverage overlaps. Compare particular model and layout combinations. |
| Activation memory | Policy-based checkpoint/swap composition and swap fusion | Selective or full recomputation, activation swap, and MoE zero-memory variants | Both cover activation memory. Their policies and scheduler interfaces differ. |
| Parallel strategy search | SAPP-ND and SAPP-PPB cost models | Core automated-pipeline and profiling tools | The SAPP search and estimator are distinct implementations. A useful port needs a MindSpeed cost-model adapter. |
| Data | Prepared Arrow and YAML-driven trainers | MCore indexed bin/idx and FSDP2 dataset processors, including an indexed-data path | The prepared-Arrow contract differs. Existing SciOWL indexed data has a MindSpeed path. |
| Checkpointing | Distributed checkpoint and async staging | MCore checkpointing and separate FSDP2 DCP and HF conversion utilities | Both have checkpoint facilities. Format and optimizer state need migration checks. |
| Profiling and recovery | Trainer callbacks and profiling tools | MCore profiling, utilization logging, and HA integrations | Existing facilities overlap. Feature combinations and recovery guarantees need separate testing. |

Source entry points:
[HyperParallel feature inventory](https://github.com/mindspore-ai/hyper-parallel/blob/c108cf94bed1c931f2a49becaf289398481745a8/docs/feature_status.md),
[optimizer guide](https://github.com/mindspore-ai/hyper-parallel/blob/c108cf94bed1c931f2a49becaf289398481745a8/docs/guide/optimizer.md),
[MindSpeed-LLM feature registration](https://github.com/YueZhou-oh/MindSpeed-LLM/blob/b82aedb6d79f9f0060e489c8aceffb3b709e568a/mindspeed_llm/features_manager/__init__.py).

## FSDP2 MegaMoE experiment

Merge these keys into the existing Qwen3 MoE FSDP2 recipe and keep the optimizer fixed when comparing dispatchers.

```yaml
parallel:
  ep_size: 16
  ep_dispatcher: megamoe
optimization:
  moe_grouped_gemm: true
```

The adapter consumes the current BF16 grouped expert weights and the existing router output. The FSDP2 model continues to own the parameters and checkpoint keys. The adapter selects the native backend's default push path. Start with an isolated EP group, TP1 and CP1. Compare with FSDP2 MC2 at the same topology, batch, and recomputation policy. Check forward outputs, hidden-state, router and expert gradients, worst-rank HBM, and checkpoint restart before running a throughput comparison. Each layer creates its own executor lazily. Measure workspace high-water memory before scaling to the full model.

The audited [native build guide](https://github.com/mindspore-ai/hyper-parallel/blob/c108cf94bed1c931f2a49becaf289398481745a8/hyper_parallel/core/multicore/docs/build.md) requires CANN toolkit/ops 9.1.0 or later, a matching Torch/torch-npu build, and an activated custom OPP payload for the target SoC. Follow the upstream [SHMEM setup](https://github.com/mindspore-ai/hyper-parallel/blob/c108cf94bed1c931f2a49becaf289398481745a8/hyper_parallel/core/multicore/docs/shmem.md) for subgroup rendezvous. The audited build requirements exclude the reported CANN 8.5.2 training environment.

## FSDP2 Asym5 Muon experiment

```yaml
training:
  optimizer: muon
  muon_ns_variant: asym5
```

The default is `legacy`. The Asym5 variant computes an update over the full logical matrix and restores its DTensor placement. This can raise temporary HBM, communication, and optimizer time because each participating shard rank materializes the matrix. Measure optimizer-step latency and peak HBM. For parameters stored as a single flattened expert matrix, this applies Muon to the stored logical matrix. Per-expert reshaping and head-wise orthogonalization are separate features. The MCore Muon implementation already has the same five coefficients under its `quintic` schedule.

Use a fresh optimizer state when switching from AdamW. When resuming a Muon run, the saved parameter groups retain the optimizer variant. Older optimizer checkpoints without a variant use legacy behavior.

Both additions are specific to `train_fsdp2.py`. The SciOWL Stage 1 MCore launcher uses `pretrain_gpt.py` and does not consume these YAML settings.
