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
    ├── deepseek-v4-flash-dspark.yaml # DeepSeek-V4-Flash + DSpark drafter recipe (NVFP4 KV, 1M ctx)
    ├── deepseek-v4-flash-dspark.env # standalone env file (mirrors the recipe env: block)
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
purpose** — several of their settings are wrong or unsafe on this homelab (see below).
**VERIFIED WORKING 2026-07-09**: live cross-node serve + inference confirmed on the
2× Spark cluster — coherent English (no fp8-KV drift), working tool calls, ~112K ctx.
The default `defaults`/`env` in the YAML ARE that verified config, so plain `make hy3`
launches it; the overrides below are only for deviating from it.

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
- **`load_format: auto` is load-bearing — NOT `instanttensor`, NOT `fastsafetensors`.**
  This is the opposite of every other 2-node recipe here, and hy3 is the exception.
  Plain `auto` (safetensors) pages shards in one at a time, keeping load-time free RAM
  at a healthy ~28 GiB/node. **`instanttensor` balloons the per-node load-time working
  set to ~121 GiB and OOM-kills the peer worker** (verified 2026-07-09); **`fastsafetensors`
  host-buffers shards → NCCL `BROADCAST` timeout / peer wedge** (2026-07-08). Both eugr
  fast-loaders lose to plain `auto` for a model this size. Load takes ~14 min.
- **Worker spawn: `VLLM_WORKER_MULTIPROC_METHOD=fork` is required.** Under vLLM's
  default `spawn`, the headless peer (node_rank 1) worker crashes at startup restoring
  a semaphore (`synchronize.py __setstate__ → FileNotFoundError` on `/dev/shm/mp-*`),
  which hangs the head at NCCL rendezvous. `fork` inherits semaphores instead. Same
  setting the mimo recipe uses. (Verified 2026-07-09.)
- **Memory/context: GMU 0.80 + fp8 KV → ~112K ctx is the verified config.** GB10 is
  unified memory (~121 GiB shared) with earlyoom, so the *load-time* footprint is what
  matters — with the `auto` loader, 0.80 holds ~28 GiB free during load and ~15 GiB at
  steady state. The forum's `0.90` wedged us; higher GMU buys nothing here since the
  loader, not GMU, governs the load footprint. `max_model_len` is `114688` (~112K):
  after the model + MTP drafter (84.5 GiB) load, ~9.4 GiB KV remains → a hard ceiling
  of **~121,648 tokens**. **Ignore `make hy3-dry` for context sizing — it's optimistic
  (~165K); 131072 fails the KV-sizing check.** Concurrency at 114688 is ~1.02x.
- **fp8_e4m3 KV — the checkpoint has no KV calibration scales**, so fp8 KV *can* cause
  **language drift** (stray Chinese in English). A 2026-07-09 smoke test showed **none**,
  but spot-check if you depend on it. fp8 is what buys ~112K ctx here (bf16 would cap
  ~60K at this GMU). Zero-drift fallback: `-o kv_cache_dtype=auto` (bf16) plus a lower
  `-o max_model_len` (~60K).
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
- **No remote reboot exists on these units** (no BMC/IPMI) — a wedged node needs the
  physical power button. That said, with this config (`auto` loader + `fork`), every
  failed attempt on 2026-07-09 died **cleanly** — the worker raised an in-process
  exception, sparkrun tore the containers down, and both nodes recovered fully with no
  wedge. The nodes have **16 GiB of swap**, and earlyoom's `-s 80` gate means it only
  SIGKILLs when swap is *also* depleted — so a brief low-MemAvailable dip during load
  does not by itself wedge the box. (The 2026-07-08 wedge was `fastsafetensors`, now
  avoided.) Still: watch free RAM on both nodes during load and `sparkrun stop` if it
  truly floors.
- **Performance** (2-Spark cross-node TP=2): forum baseline ~21.8 tok/s single stream,
  ~59.7 tok/s aggregate 6-way. Local 2026-07-09 smoke test: coherent output at
  ~14.5 tok/s end-to-end on a short single-stream request (incl. prefill).
- Keep `hy3-295b-nvfp4.env` in sync with the `env:` block in the YAML if you edit
  either.

## deepseek-v4-flash-dspark — caveats

Community recipe from
[tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark),
serving **DeepSeek-V4-Flash** across both Sparks with the **DSpark** speculative-decoding
drafter and the experimental **`nvfp4_ds_mla`** KV cache at **1M context**. It is
**unverified on this host**. `make deepseek-dspark` launches it; `make deepseek-dspark-dry`
estimates fit first.

> **Rewritten 2026-07-15.** This recipe previously followed eugr's forum thread
> (fp8 KV, 256K ctx, `vllm-node` image). It now follows tonyd2wild's repo: NVFP4 KV,
> 1M ctx, Stage C image. The old fp8 lane is gone — `git log recipes/` to recover it.

> **Not the same as `make deepseek`.** `make deepseek` runs the published
> `@experimental/deepseek4-flash-fp8-mtp-vllm` (fp8 + **MTP** drafter). This recipe is
> the separate **DSpark**-drafter variant, kept as a local YAML.

- **2 Sparks required** (`cluster_only`, `min_nodes: 2`). Cross-node
  `tensor_parallel: 2`, one GB10 per node. It cannot run solo.
- **Which checkpoint? Upstream contradicts itself — we pin `fraserprice/…`.** The
  compose/`.env` default is `deepseek-ai/DeepSeek-V4-Flash-DSpark`, but the upstream
  README's "Weights" section and *every* verified deployment log load
  [`fraserprice/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark)
  (HF tags it `base_model:quantized:deepseek-ai/DeepSeek-V4-Flash`; the model path in
  the logs is `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`). The
  `nvfp4_ds_mla` KV path was validated against fraserprice's checkpoint, so that is what
  the recipe pins. **Both repos exist and neither is gated** — if you switch to the
  `deepseek-ai` one, expect a possible weights/KV-dtype mismatch with the Stage C image.
- **The container is a *locally built* 3-stage image, not a published tag.** The recipe
  points at **`vllm-dspark-runtime:dspark-nvfp4-stage-c`**. Build it on the head with
  upstream's script — it rsyncs and rebuilds on `WORKER_HOST` by default, so both nodes
  get it:
  ```bash
  git clone https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark
  cd DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark
  cp .env.dspark.example .env.dspark   # set WORKER_HOST / MASTER_ADDR / NCCL_IB_HCA
  ./build-dspark-vllm-runtime.sh       # base overlay -> NVFP4 stage A -> B -> C
  ```
  This is a **long build** (it compiles NVFP4 dtype support and the padded KV envelope
  for DeepSeek-V4 sparse MLA). Nothing in this repo builds it for you.
- **The `dspark_proposer.py` bind-mount is deliberately dropped.** Upstream's compose
  mounts `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py` over site-packages to
  carry the concurrency fix (Keys Patch 2b / Patch 3). A sparkrun recipe **cannot mount
  that overlay** — but per upstream's README, *a fresh Stage C build already contains
  Patch 3* (commit `e83606a`), so the mount is unnecessary. **If you reuse an older
  pre-Patch-3 image** (e.g. `probe-c-p2b`), this recipe will run **without** the
  concurrency fix — rebuild Stage C rather than working around it.
- **`nvfp4_ds_mla` KV + block-size 256 + 1M context.** `kv_cache_dtype: nvfp4_ds_mla`
  exists **only in the Stage C image** — it is what buys 1M ctx. `max_model_len: 1048576`
  is the model's *true* YaRN ceiling (`original_max_position_embeddings` 65536 x factor
  16). Upstream benchmarks 1.5M via `VLLM_ALLOW_LONG_MAX_MODEL_LEN`, but that
  extrapolates past calibration — it boots and benchmarks, yet **coherent output past 1M
  is not guaranteed**. The published 1.5M numbers are "how far it was pushed", not a
  quality claim.
- **`make deepseek-dspark-dry` can NOT vet the KV sizing on this recipe.** sparkrun
  0.2.39 doesn't know the dtype and says so: `Warning: Unknown KV cache dtype
  'nvfp4_ds_mla'`. Its "Available for KV" figure therefore assumes a dtype this recipe
  isn't using, so a too-large `max_model_len` will **not** be caught. The dry run is
  still useful for the weights half — it reports `Model weights: 155.43 GB`,
  `Per-GPU total: 77.71 GB`, `DGX Spark fit: YES` (verified 2026-07-15) — but the 1M
  context claim rests on upstream's reported 3,225,280-token KV pool, not on anything
  checked locally.
- **We take the conservative 1M lane, not upstream's recommended one.** Upstream's
  `.env.dspark.example` recommends `max_num_seqs=12` / `GMU=0.85`; this recipe uses
  **`max_num_seqs: 6` / `GMU: 0.80`** — upstream's own "conservative prior agent lane",
  and the profile of its **2026-07-04 verified 1M deployment**. Reason: `0.85` sits
  exactly at this homelab's earlyoom ceiling (earlyoom SIGKILLs the worker if the
  load-time page-cache spike drops free RAM under threshold). Context and concurrency
  share one KV pool, so 6 slots at 1M is the safe pairing. Bump with
  `make deepseek-dspark GPU_MEM=0.85` if you want upstream's throughput lane.
- **`max_cudagraph_capture_size: 24` is derived, not arbitrary.** It must equal
  `max_num_seqs * (num_speculative_tokens + 1)` = `6 * (3 + 1)`. Upstream computes it in
  shell; sparkrun has no template arithmetic, so it is a literal. **If you change
  `max_num_seqs`, recompute it** — capture-size-matches-batch is part of the garble fix.
- **DSpark spec decode: 3 tokens, probabilistic draft.** The 2026-07-03 **garble fix**:
  greedy draft + a 5-token window + uncaptured graphs corrupted tool calls on first
  prompts under concurrent load. Fixed by `num_speculative_tokens: 3`,
  `draft_sample_method: probabilistic`, and the capture-size rule above. These live
  inside the literal JSON `--speculative-config` blob — sparkrun (0.2.39) doesn't
  substitute `{placeholders}` nested in literal braces, so they are **hardcoded** in the
  `command:` (same for the `--reasoning-config` and `--default-chat-template-kwargs`
  blobs). Upstream reports ~50-60 tok/s single-stream and ~230 tok/s aggregate over 12
  streams at 60.2% acceptance.
- **`VLLM_USE_B12X_MOE=1` is the entire speed difference.** `=1` selects the b12x Mxfp4
  MoE backend (boot log `Using 'B12X' Mxfp4 MoE backend`); `=0` **silently** falls back
  to `DEEPGEMM_MXFP4` and tanks decode to ~29 tok/s. Three things upstream explicitly
  forbids on this image, all omitted here: `VLLM_USE_B12X_FP8_GEMM=1` (DeepGEMM layout
  assert during drafter warmup), `VLLM_USE_V2_MODEL_RUNNER=1` (hard-rejected with DSpark
  spec decode), and `--attention-backend FLASHINFER_MLA_SPARSE_DSV4` (that backend name
  doesn't exist on this image — leave attention AUTO).
- **No `--override-generation-config`.** Removed in the garble fix: it carried
  `repetition_penalty=1.05`, a documented DSpark spec-decode crash risk (illegal memory
  access). The recipe passes `--generation-config vllm` only; explicit client request
  params still win.
- **`HF_TOKEN` is optional here.** The checkpoint is **public/non-gated** (the old forum
  recipe's "gated model" note was wrong). A token only helps with anonymous rate limits.
- **No `--distributed-executor-backend` / `--nnodes` family.** Upstream's compose
  hand-rolls `mp` + `--nnodes 2 --node-rank --master-addr --master-port --headless` per
  node, because it launches raw docker per node. sparkrun's `vllm-distributed` runtime
  injects those itself and picks a compatible backend (`mp` is vLLM's default for
  `nnodes>1` anyway). The recipe omits them all, matching every other 2-node recipe here.
- **Port 8000, not upstream's 8888.** Repo convention. Upstream's smoke/bench scripts
  assume `:8888` — pass `-o port=8888` if you want to run them unmodified.
- **CUDA arch pinned to `12.1a`** for GB10 Blackwell (repo convention, build-time). The
  NCCL block mirrors upstream's, except `NCCL_IB_GID_INDEX=3` (homelab value; upstream's
  example says `0`) and the per-node `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` / `VLLM_HOST_IP`,
  which are left to sparkrun. Adjust if cross-node all-reduce misbehaves.
- Keep `deepseek-v4-flash-dspark.env` in sync with the `env:` block in the YAML if you
  edit either.

## qwen3.6-35b-a3b-nvfp4-fast — caveats

Unsloth's Spark-targeted quant of Qwen3.6-35B-A3B (released 2026-07-10), added
2026-07-12 from the [Unsloth DGX Spark instructions](https://unsloth.ai/docs/models/qwen3.6).
Launch: `make qwen35` (single node, TP=1); fit-check: `make qwen35-dry`.

- **Status: dry-run validated only.** Both Sparks were serving deepseek-dspark
  when this was added, so it has not had a live boot on this host yet. First
  launcher: watch for the two failure modes flagged below (b12x rejection,
  MTP-draft backend).
- **`flashinfer_b12x` is load-bearing — and sm_121-specific.** Per Unsloth, GB10
  must force `--moe-backend/--linear-backend flashinfer_b12x` for this quant or
  serving silently degrades to Marlin W4A16 (~2.5x slower — that is what the
  older `@eugr/qwen3.6-35b-a3b-nvfp4` recipe runs on the `nvidia/` checkpoint).
  This is the **opposite** of the sm_120 (RTX 5090) rule, where forcing a MoE
  backend is what causes the 2.5x loss. `eugr/spark-vllm:latest` (vLLM 0.23.1
  dev790) passes Unsloth's b12x preflight on both nodes (verified 2026-07-12);
  the preflight command is in the recipe header.
- **1M context via static YaRN.** Default `max_model_len` is 1,010,000 using the
  Qwen model card's exact `--hf-overrides` rope blob (factor 4.0) +
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. Memory is comfortable on one GB10
  (~22 GiB weights + ~10 GiB KV at 1M; KV is cheap — 10/40 full-attn layers,
  2 KV heads). **Trade-off:** static YaRN slightly degrades short-context
  quality (Qwen's own warning — the scale applies at every length). For mostly
  ≤256K work, drop the `--hf-overrides` line and `-o max_model_len=262144`; for
  ~512K, set `"factor": 2.0` in the blob.
- **fp8 KV is calibrated here** — unlike hy3, this checkpoint ships KV scales,
  so the language-drift risk of uncalibrated fp8 KV does not apply. Keep
  `kv_cache_dtype: fp8`.
- **MTP spec decode: K=2** (Unsloth's recommendation; their fixed MTP head).
  Unverified acceptance on GB10 for this quant — the same model's pos-2
  acceptance measured ~0.72 on the 5090 box. If the b12x MoE backend rejects
  the draft module at startup, add `"moe_backend":"triton"` inside the
  `--speculative-config` JSON (eugr's trick on the Marlin recipe). If accepted
  throughput looks bad, drop to `"num_speculative_tokens":1` (hy3 lesson: on
  GB10, pos-2 acceptance can make K=2 a net loss).
- **No chat-template mod needed.** Unsloth checkpoints ship a fixed chat
  template + fixed MTP head — do not apply `mods/fix-qwen3.6-chat-template`
  (that mod exists for the `Qwen/` originals).
- **Higher-precision alternatives on this hardware** (all fit on one GB10, all
  bandwidth-bound so roughly proportional decode cost): `Qwen/Qwen3.6-35B-A3B-FP8`
  (~37 GiB, ~2x the active-weight reads → existing
  `@official/qwen3.6-35b-a3b-fp8-mtp-vllm` / `@eugr/qwen3.6-35b-a3b-fp8`
  recipes, also 1M-capable with the same YaRN blob) and BF16 (~70 GiB, ~4x
  reads). This NVFP4-Fast recipe is the speed end of that curve; the FP8
  official recipes are the quality end.
