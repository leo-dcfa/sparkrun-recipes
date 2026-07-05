# Leo's local sparkrun recipes

Local recipe registry for the DGX Spark homelab (2x GB10, nodes `spark-f31f`
`169.254.225.22` + `spark-d306` `169.254.17.50`, linked over ConnectX-7).

```
sparkrun-recipes/
├── .sparkrun/registry.yaml          # registry manifest (name: local)
└── recipes/
    ├── mimo-v2.5-dflash.yaml        # MiMo-V2.5 NVFP4 + DFlash recipe
    ├── mimo-v2.5-dflash.env         # standalone env file (mirrors the recipe env: block)
    ├── ornith-1.0-397b.yaml         # Ornith-1.0-397B INT4 (W4A16 AutoRound) recipe
    ├── ornith-1.0-397b.env          # standalone env file (mirrors the recipe env: block)
    └── mods/                        # recipe mods (empty for now)
```

## Running

By file path (works immediately, no registration):

```bash
sparkrun run sparkrun-recipes/recipes/mimo-v2.5-dflash.yaml \
  --tp 2 --hosts 169.254.225.22,169.254.17.50
```

By name (after registering this dir as a registry — see below):

```bash
sparkrun run mimo-v2.5-dflash --tp 2 --cluster <your-2node-cluster>
```

## Registering as a local registry

This folder is a git repo with a `.sparkrun/registry.yaml` manifest, so sparkrun
can treat it like any other registry (git can clone from a local path):

```bash
sparkrun registry add /home/leo/sparkrun-recipes
sparkrun registry update
sparkrun list | grep -iE 'mimo|ornith'
```

To remove it later: `sparkrun registry remove local`.

> **DeepSeek-V4-Flash lives elsewhere now.** The local `deepseek-v4-flash` recipe
> was removed — the homelab runs the published `@experimental/deepseek4-flash-fp8-mtp-vllm`
> recipe instead (see `make deepseek` in the Makefile), so keeping a divergent local
> copy was just a maintenance hazard.

## mimo-v2.5-dflash — caveats

Community recipe from
[DoctorMasterNewb/vLLM-Mimo-V2.5-Dflash-2x-DGX-Spark](https://github.com/DoctorMasterNewb/vLLM-Mimo-V2.5-Dflash-2x-DGX-Spark):
Xiaomi **MiMo-V2.5** (`lukealonso/MiMo-V2.5-NVFP4`, NVFP4 weights, ~170 GB)
served across both Sparks with a **DFlash block-diffusion drafter** for
speculative decoding. It is **unverified on this host**. Before relying on it:

- **2 Sparks required** (`cluster_only`, `min_nodes: 2`). Cross-node
  `tensor_parallel: 2`, one GB10 per node. It cannot run solo.
- **No canonical container image.** Upstream builds vLLM **≥ 0.23.1 nightly**
  (validated `0.23.1rc1.dev537`, 2026-06-28, with the `dflash` spec method
  registered) from the `eugr/spark-vllm-docker` base, applies **four MiMo/DFlash
  patches** (MiMo-V2 config registration, NVFP4 fused-QKV load fix, vision-merger
  packed-weight load, and PR #46104 for SWA+DFlash on the Triton backend), and
  bakes the chat template into the image. It also needs **PR #45181** (mixed KV
  page sizes) and `transformers` 5.x. The `container:` tag in the recipe
  (`eugr/spark-vllm-docker:mimo-v2.5-dflash-nightly`) is a **placeholder** —
  build/push your own image from that repo and swap the tag in.
- **Pre-download the drafter on BOTH nodes.** `XiaomiMiMo/MiMo-V2.5-DFlash`
  (~2.9 GB) must be local to each TP rank. Upstream:
  `scripts/download-drafter.sh ~/.cache/huggingface/dflash-mimo-v2.5`. The recipe
  points `--speculative-config` at `defaults.drafter_dir`
  (`/root/.cache/huggingface/dflash-mimo-v2.5/dflash`) — make sure that path
  resolves to the downloaded drafter inside the container on each node, or
  override `-o drafter_dir=…`.
- **Tight KV memory headroom.** 170 GB weights + drafter + profiling peak at
  `gpu_memory_utilization: 0.83`; `max_model_len` is 131072 with bf16 KV. Don't
  push GMU much higher.
- **First cold load is slow.** ~170 GB is pulled/loaded at serve time — upstream
  wraps the launch in systemd with `TimeoutStartSec=3600`. If sparkrun has a
  startup timeout, give it plenty of room.
- **CUDA arch pinned to `12.1a`** for GB10 Blackwell. `TORCH_CUDA_ARCH_LIST` is
  from upstream; `FLASHINFER_CUDA_ARCH_LIST` is added here for repo parity.
- **NCCL knobs differ from the Ornith recipe on purpose** — MiMo uses the
  upstream author's tuning (`NCCL_PROTO=LL`, `NCCL_MAX_NCHANNELS=2`,
  `NCCL_NET_GDR_LEVEL=LOC`, `NCCL_NVLS_ENABLE=0`), not the standard IB settings
  in `ornith-1.0-397b.env`.
- **Multi-node flags omitted / no explicit ray backend.** Upstream `serve.sh` is
  a plain `vllm serve`; sparkrun's `vllm-distributed` runtime handles head/worker
  coordination. If cross-node TP=2 won't initialize, add
  `--distributed-executor-backend ray` to the `command:` block (as
  `ornith-1.0-397b.yaml` does).
- **Speculation payoff is workload-dependent.** Baseline ~22 tok/s (single
  stream, cross-node TP=2); with DFlash ~22–67 tok/s — structured/tool output
  accepts 6+ drafts, free-form prose at nonzero temperature is the worst case
  (~1.5×).
- Keep `mimo-v2.5-dflash.env` in sync with the `env:` block in the YAML if you
  edit either.
