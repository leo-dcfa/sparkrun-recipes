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
    ├── hy3-295b-nvfp4.yaml          # Hy3-295B (Hunyuan 3) NVFP4-W4A16 + MTP recipe
    ├── hy3-295b-nvfp4.env           # standalone env file (mirrors the recipe env: block)
    └── mods/                        # recipe mods (empty for now)

docker/
└── Dockerfile.hy3                  # builds the local eugr/spark-vllm:hy3-opensource image
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

> **MiniMax-M3 (REAP-25) is a published experimental recipe.** `make m3` runs
> `@experimental/minimax-m3-v0-nvfp4-2x-reap25` — MiniMax **M3 v0** in NVFP4,
> 25%-expert-pruned with Cerebras [REAP](https://arxiv.org/abs/2510.13999)
> (`sparkarena/Minimax-M3-v0-NVFP4-REAP25`, ~93.5B params / 43.5 GB weights). Note it
> runs on the **sglang** runtime (not vLLM) via the `scitrera/dgx-spark-sglang-mm:v0`
> container, cross-node **TP2**. Like the DeepSeek recipe it lives in the published
> `experimental` registry, so there's no local YAML here — inspect it with
> `sparkrun show @experimental/minimax-m3-v0-nvfp4-2x-reap25`. It's **experimental /
> unverified on this host**: the default `max_model_len` is a conservative `32768`
> (fit estimate shows ~40× headroom, up to ~1.33M tokens), and GMU defaults to `0.81`
> — at/under the homelab `0.85` ceiling, so leave it there. Sister variants exist in
> the same registry: `…-2x-reap50` (50%-pruned) and `…-4x` (unpruned, 4-node). Run
> `make m3-dry` before launching.

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

## hy3-295b-nvfp4 — caveats

Community recipe from the NVIDIA developer forums thread
[Hy3-295B NVFP4-W4A16 + MTP on 2× DGX Spark](https://forums.developer.nvidia.com/t/hy3-295b-hunyuan-3-nvfp4-w4a16-mtp-speculative-on-2x-dgx-spark-gb10-128k-ctx-21-8-tok-s-single-59-7-tok-s-6-way/375851),
serving Tencent **Hy3 / Hunyuan 3** (295B total, 21B active MoE, 192 experts top-8,
3.8B MTP layer; Apache-2.0, released **2026-07-06**) requantized to weight-only
**NVFP4 (W4A16)** as [`kodelow/Hy3-NVFP4-W4A16`](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16)
(~168 GiB / 99 shards). This recipe **diverges from the forum/tonyd2wild version on
purpose** — three of their settings are wrong or unsafe on this homelab (see below).
Verified so far: download, image, parser fix, cross-node NCCL rendezvous. **A live
serve + inference is not yet confirmed** — treat it as unverified until you run the
runbook below.

- **Only the 4-bit quant fits.** Hy3 ships BF16 (~598 GB) and `tencent/Hy3-FP8`
  (~295 GB) — **neither fits** the two Sparks' ~243 GB combined usable memory. This
  weight-only NVFP4-W4A16 build (~168 GiB: routed experts 4-bit E2M1 with FP8 group
  scales; shared expert, attention, router, dense MLPs, embeddings, `lm_head`, norms
  stay BF16) is what makes it fit at cross-node TP=2.
- **2 Sparks required** (`cluster_only`, `min_nodes: 2`). Cross-node
  `tensor_parallel: 2`, one GB10 per node. **TP=3 is impossible** (8 KV heads don't
  divide by 3) — this is a 2-Spark (or 4-Spark) recipe. It cannot run solo.
- **The container is a *locally built* image, not a published tag.** The recipe
  points at **`eugr/spark-vllm:hy3-opensource`**, which is `eugr/spark-vllm:latest`
  (already ships vLLM 0.23.1 dev790 + the HYV3 model + the layer-80 MTP module + the
  `hy_v3` parsers) with a one-line patch. Build it from the committed
  [`docker/Dockerfile.hy3`](docker/Dockerfile.hy3); sparkrun distributes the local
  image to both nodes (it is not pushed to any registry).
- **Why the patch:** the `hy_v3` parsers derive their special-token suffix from
  `tokenizer init_kwargs["token_suffix"]`, but this checkpoint's
  `tokenizer_config.json` **does not set it**, so the parsers fall back to bare
  `<think>`/`<tool_call>` and **crash at startup** ("could not locate think
  start/end tokens"). The fix defaults that suffix to `:opensource`. **tonyd2wild's
  `sed` patch does NOT apply** — its literal bare-token strings don't exist in this
  newer parser; patch the default in the parser source instead (see the Dockerfile).
- **`load_format: instanttensor` is mandatory here — never `fastsafetensors`.** On
  2026-07-08 a run with `--load-format fastsafetensors` host-buffered shards, spiked
  free RAM to ~500 MiB **during weight load**, tripped earlyoom → a rank was killed →
  the survivor hung on the next NCCL `BROADCAST` (600 s timeout) → **the peer wedged
  and needed a physical power-cycle**. `instanttensor` (eugr's streaming loader, used
  by every other 2-node recipe here) keeps the load-time peak low and is the single
  most important safety setting in this recipe.
- **Memory/context: GMU 0.85 + bf16 KV → ~115K ctx is the safe default.** GB10 is
  unified memory (~121 GiB shared CPU+GPU) with earlyoom running, so a too-high GMU
  or a load-time spike gets the vLLM worker SIGKILLed. The forum runs `0.90`; **0.90
  OOM-wedged the peer here.** `0.85` is the homelab fleet ceiling (the `minimax`
  M2.7 NVFP4 recipe runs it), leaves ~18 GiB free RAM at steady state, and is safe
  *because* instanttensor
  removes the load spike. **Do not exceed 0.85.** `max_model_len` is `114688`
  (~112K) — confirm it fits with `make hy3-dry` before launching.
- **bf16 KV on purpose.** The checkpoint ships **without KV calibration scales**, so
  uncalibrated fp8 KV can cause **language drift** (occasional Chinese output in
  English contexts). The recipe defaults `kv_cache_dtype: auto` (bf16). Only set
  `-o kv_cache_dtype=fp8` on a KV-calibrated revision — and spot-check quality if
  you do. (fp8 KV would roughly double the context ceiling.)
- **Spec decode: `num_speculative_tokens: 1` only.** The layer-80 MTP drafter hits
  62–76% pos-1 acceptance (83% on GSM8K), but pos-2 acceptance is only ~20% on GB10,
  so **spec-2 is a ~30% throughput LOSS.** It is hardcoded to 1 inside the JSON
  `--speculative-config` blob (sparkrun 0.2.39 doesn't substitute placeholders nested
  in literal `{...}`). Don't raise it.
- **`--enforce-eager` + `--moe-backend marlin` are load-bearing.** FlashInfer's
  native FP4 path freezes GB10s, so CUDA graphs are disabled; the MARLIN NvFp4 kernel
  serves this weight-only scheme (the FlashInfer W4A4 backends reject it).
- **No `--distributed-executor-backend ray`.** sparkrun's `vllm-distributed` runtime
  injects vLLM's native multi-node flags (`--nnodes N --node-rank R --master-addr/
  --master-port`, one vllm per node) and vLLM 0.23 **rejects `ray` together with
  `nnodes>1`**. The forum's `ray` is for a hand-rolled ray cluster, not sparkrun.
- **CUDA arch pinned to `12.1a`** for GB10 Blackwell (repo convention, build-time).
  NCCL knobs are the homelab's standard IB settings (the forum specified only ray +
  ConnectX-7 MTU 9000). Adjust if cross-node all-reduce misbehaves.
- **No remote reboot exists on these units** (no BMC/IPMI). If a launch wedges a
  node, only the physical power button recovers it — which is why the memory-safety
  settings above are non-negotiable. Run `make hy3-dry` first, and watch free RAM on
  both nodes during load.
- **Performance baseline** (forum, 2-Spark cross-node TP=2, 128K ctx): ~21.8 tok/s
  single stream, ~59.7 tok/s aggregate 6-way concurrent.
- Keep `hy3-295b-nvfp4.env` in sync with the `env:` block in the YAML if you edit
  either.
