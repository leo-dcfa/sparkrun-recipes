# Makefile — convenience wrappers around `sparkrun run` for the recipes Leo runs
# day-to-day on the 2x DGX Spark (GB10) homelab.
#
# make deepseek                         # launch DeepSeek-V4-Flash-Vision-Exp + DSpark MTP=6, NVFP4 KV, 1M ctx (MiaAI-Lab kit, NOT sparkrun)
# make deepseek-sparkrun                # ROLLBACK lane: the tonyd2wild sparkrun recipe this replaced (k=5, 12 seqs)
# make ds41                             # launch DeepSeek-V4.1-Flash EXL3 2.9bpw (MiaAI-Lab kit, 552B, text-only, 600K ctx) -- NEEDS WEIGHTS, see below
# make glm-exl3                         # launch GLM-5.3-Flash EXL3 4bpw + DFlash2 k=7, 850K ctx (MiaAI-Lab kit, NOT sparkrun) — the GLM 5.3 lane
# make qwen38fn                         # launch Qwen3.8-Flash-Next NVFP4 (MiaAI kit, nvidia ckpt, fp8 KV, YaRN 1M ctx) — the qwen-flash lane
# make qwen-flash                       # launch Qwen3.8-Flash-Next NVFP4 (nvidia ckpt), tonyd2wild vLLM TP2 SPEED lane, 262K ctx, thinking ON (2-node, NOT sparkrun)
# make qwen-flash-no-thinking           # same lane, enable_thinking false server-side
# make mimo                             # launch MiMo-V2.6-Flash-RL (Xiaomi 309B-A15B omni, MiaAI-Lab SGLang TP2/EP2 kit, MXFP4 experts, fp8 KV, 1M ctx, DFlash k=8; NOT sparkrun)
# make mimo-vllm                        # ROLLBACK lane: tonyd2wild's vLLM TP2 + DFlash k=7 kit this replaced (300K ctx, thinking OFF server-side)
# make deepseek MAX_MODEL_LEN=500000    # override context length
# make deepseek-dry                     # VRAM/fit estimate, no launch
# make stop                             # stop everything on the cluster
#
# Recipes are registry-qualified (@registry/name); resolved via the enabled
# sparkrun registries — no local checkout needed.

SPARKRUN ?= sparkrun
CLUSTER  ?= leo-azl-2node

# sparkrun lives in ~/.local/bin, which non-interactive shells (e.g.
# `ssh spark-f31f make -C sparkrun-recipes deepseek`) don't have on PATH — without
# this, make dies with the unhelpful "make: sparkrun: No such file or directory".
export PATH := $(HOME)/.local/bin:$(PATH)

# Registry-qualified recipe identifiers.
# (Pre-0731 DeepSeek lanes retired 2026-08-19 — the old arena recipe and
#  `make deepseek-dspark` + its MiaAI compose fallback; 0731 is the only DeepSeek
# lane now. Recipes archived in git history.)
# (MiniMax-M2.7 retired 2026-08-29 — was @official/minimax-m2.7-nvfp4-vllm; Leo stopped using it.)
# (MiniMax-M3 REAP25 retired 2026-07-24 — was @experimental/minimax-m3-v0-nvfp4-2x-reap25.)
# (Spark Qwen lanes retired 2026-07-24 — were @official/qwen3.6-27b-fp8-mtp-vllm
# and recipes/qwen3.6-35b-a3b-nvfp4-fast.yaml; the yaml stays in recipes/ for reference.
#  Qwen3.8-Flash-Next came back 2026-09-03 as `make qwen38fn`, see below.)
# (Hy3-295B, MiMo-V2.5 Omni, Inkling-Small (+ the inkling-eugr spark-vllm-docker
#  fallback) and Step-3.7-Flash retired 2026-09-04 — Leo stopped using them.
#  Were recipes/hy3-295b-nvfp4.yaml, recipes/mimo-v2.5-omni.yaml,
#  recipes/inkling-small-nvfp4.yaml (+ ~/spark-vllm-docker) and
#  recipes/step-3.7-flash-nvfp4.yaml; the yamls stay in recipes/ for reference,
#  the targets are in git history. LiteLLM entries removed the same day.)

# Local recipes (this repo) — run by file path, no registry needed.
# DeepSeek-V4-Flash-Vision-Exp (native image input) + DSpark — tonyd2wild's
# vision port, adopted 2026-09-01 from the vision-exp-default branch of the
# (renamed) upstream repo. Same LM as 0731 with a 3-layer drafter (k=5)
# and the ViT+aligner running inside vLLM; image vllm-dspark-runtime:
# dspark-nvfp4-vision-exp is built on BOTH nodes with
# docker/Dockerfile.dspark-vision-exp, then -p5 on top of it with
# docker/Dockerfile.dspark-vision-exp-p5. Re-synced 2026-09-03 to upstream main:
# k=5 (their k=3 A/B was measured without Patch 4 and is retracted), an
# inference-time RPC deadline, and Patch 5 (stop strings no longer truncate
# reasoning). See the yaml header for the full adoption notes and deviations.
# (The 0731 TEXT lane was superseded the same day; its yaml + p4a image are
#  kept for rollback — point DEEPSEEK_RECIPE back at
#  recipes/deepseek-v4-flash-0731.yaml. It is the faster lane for text-only
#  work: ~64 tok/s battery mean vs the vision lane's upstream ~55/33.)
DEEPSEEK_RECIPE       := recipes/deepseek-v4-flash-vision-exp.yaml
# SWITCHED 2026-09-12 at Leo's request: `make deepseek` is now MiaAI-Lab's DSpark
# kit (github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark @ f3d7645,
# clone + this pair's .env.dspark at ~/src/miaai-ds4-dspark). The tonyd2wild
# sparkrun recipe it replaced is kept as `make deepseek-sparkrun` — it is NOT
# retired, just demoted, and DEEPSEEK_RECIPE above still drives it.
#
# Why the switch is more than a re-skin: the kit carries ~25 boot-time hotfixes
# the recipe does not (issue 22 nvfp4_ds_mla long-context decode, issue 27
# partial-prefill concurrency, issue 55 tool truncation, issue 117 shm ring
# buffer, issue 43 decode fairness, issue 26 hybrid SWA min, issue 133 triton
# specialization, the MTP buffer / skip-topk / dense-prefill-indexer /
# flashmla-workspace / grammar-advance set, plus GB10 spin-wait). It also pins
# the runtime by DIGEST (anemll 0.1.1 @ sha256:a8394849) rather than tracking a
# locally built image.
#
# Serving-shape deltas vs the sparkrun lane: MTP k=6 (not DSpark k=5),
# MAX_NUM_SEQS 6 (not 12), gmu 0.835 (not 0.85), --long-prefill-token-threshold
# 1024, --moe-backend flashinfer_b12x, DEFAULT_THINKING=low (the recipe served
# thinking:false). Same checkpoint (deepseek-ai/DeepSeek-V4-Flash-Vision-Exp,
# already cached — nothing re-downloads), same 1M ctx, same nvfp4_ds_mla KV,
# same served name deepseek-v4-flash-vision-exp and port 8000, so the LiteLLM
# entry and both Mac client lists are unchanged.
#
# LOCAL DEVIATIONS in .env.dspark, all upstream-supported optional overrides:
# WORKER_NCCL_IB_HCA / WORKER_{NCCL,TP,GLOO}_SOCKET_IFNAME pin the worker to its
# f1 port (this pair is CROSS-WIRED head f0 / worker f1; upstream's single
# TP_SOCKET_IFNAME assumes both nodes match, which would strand the worker).
# start-deepseek-v4-flash-dspark.sh:581-584 is where they are read.
#
# 2026-09-12: pulled c444d70 -> f3d7645 and adopted the 16 keys upstream added
# since this .env was written, at upstream defaults. All ship 0 except
# DSPARK_ASYNC_SCHEDULING=1, so that adoption changed no behaviour; the block is
# marked LEO 2026-09-12 in .env.dspark and lists what each knob does. Several
# (ROPE_SWA_FIX, DSML_RECOVERY, C128A_PREFILL_CACHE, DSPARK_BLOCK_K, SWA_PREFIX,
# MXFP4_INDEXER_CACHE) are experimental opt-ins — turn on ONE at a time with a
# measurement, never as a batch.
DEEPSEEK_MIAAI_DIR    := $(HOME)/src/miaai-ds4-dspark
# DeepSeek-V4.1-Flash EXL3 2.9bpw -- ADDED 2026-09-13, a SECOND DeepSeek lane.
# It does NOT replace `make deepseek`: that stays the V4-Flash Vision-Exp lane
# and remains the only vision-capable DeepSeek here (this one is text-only,
# LANGUAGE_MODEL_ONLY=1 upstream). Kit: MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-
# DGX-Sparks @ 8404ac7 (pulled 2026-09-17: docs, .env.example, opt-in cooperative MoE), clone + this pair's .env at ~/src/ds41-exl3-miaai
# (local values marked LEO: in that .env).
#
# Why this lane is interesting: V4.1-Flash is 552B (vs V4-Flash's 284B) and
# until now needed FOUR Sparks. Two things make 2 nodes work -- EXL3 at an
# average 2.9 bpw (mul1 codebook, per-tensor K: routed experts K=3 except
# layers 18-22 at K=2, shared experts K=5/4), and the ~190 GiB Engram n-gram
# tables being served FILE-BACKED over NFS/ZFS instead of resident in the GPU
# pool. Upstream on 2x GB10: 31.6 tok/s single stream, 42.5 aggregate at x2,
# ~1,041 tok/s prefill at 10k, 810 tok/s on a 601k prompt (742 s TTFT).
# For contrast the only other 2x V4.1 recipe (sfxnz) is 2.0 bpw / 21.2 tok/s,
# and 2.0 bpw is below the bitrate where EXL3 quality falls off a cliff on a
# comparable MoE, so 2.9 is the materially better lane.
#
# NOT RUNNABLE UNTIL THE WEIGHTS ARE FETCHED -- ~387 GiB, nothing cached:
#   cd $(DS41_DIR) && ./download.sh          # 197 GiB EXL3 + 190 GiB Engram
#   ./start.sh share                         # export Engram over NFS (default)
#   ./start.sh pack                          # ...or local NVMe, +25-50% prefill
# Then `make ds41`. First boot is ~25 min.
#
# HEADROOM: upstream budgets ~116 GiB of a 121 GiB node and notes long prompts
# reaching a 2.1 GiB MemAvailable floor. earlyoom here is -m 2 (~2.4 GiB) with
# --prefer vllm, so this sits CLOSER to the kill line than GLM-EXL3 at 850k
# (~2.9 GiB). Watch the first long prefill; `flush` is a prerequisite below.
DS41_DIR              := $(HOME)/src/ds41-exl3-miaai
# (GLM-5.3-Flash NVFP4 retired 2026-09-14 at Leo's request — `make glm`, the
#  tonyd2wild DFlash2 k=7 lane on recipes/glm-5.3-flash-dflash2.yaml (RedHatAI
#  W4A4 ckpt, 256K ctx, vision + thinking on). Its LiteLLM entry glm-5.3-flash
#  went the same day; `make glm-exl3` is now the only GLM 5.3 lane. Both recipe
#  yamls are KEPT for rollback: glm-5.3-flash-dflash2.yaml and the older MTP-4
#  glm-5.3-flash-nvfp4.yaml (MiaAI v8 image glm53-flash-sm121:v8, LibertAIDAI
#  weights). To roll back: restore GLM_RECIPE + the glm/glm-dry/stop-glm targets
#  from git history and re-add the LiteLLM entry.)
# GLM-5.3-Flash EXL3/TR3 4bpw + DFlash2 k=7 — the GLM 5.3 lane (it was the A/B
# partner of `make glm`, NVFP4, retired 2026-09-14).
# SWITCHED 2026-09-10 from Reederey87's fork to MiaAI-Lab's ORIGINAL kit at
# Leo's request (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks @
# 94bddea, clone + this pair's .env at ~/src/glm53-exl3-miaai; every local value
# is marked LEO: in that .env). PULLED FORWARD to 1caea9a on 2026-09-11.
# Upstream's only runtime-visible change in that range: the launcher now lets
# ANY exported caller variable win over .env (it snapshots `compgen -e` instead
# of a fixed whitelist), so the `set -a && . ./.env` in the launch command below
# is now redundant — not harmful, it re-supplies the identical values. Nothing
# under Dockerfile/overlay/files/ablit moved, so the image is unchanged in
# substance; but `tests/` IS in start.sh's recipe stamp and one changed test
# (tests/test_indexer_workspace.py) is COPYed at Dockerfile:462 of 495, so the
# first launch after the pull rebuilds only the tail off the layer cache
# (minutes, not the fork's ~40 min). SKIP_BUILD=1 skips it and warns about the
# stamp instead. GHCR's :exl3 is itself stale (built 2026-09-07, stamp
# 825e3374), so pulling the image would not settle the stamp either.
#
# 2026-09-13: PULLED 1caea9a -> f906ee9 (30 commits). Mostly launcher hardening:
# RoCE GID now validated on every listed CX7 HCA, PYTORCH_CUDA_ALLOC_CONF
# overridable, jinja2 host-python lookup for chat-template validation,
# comma-separated dual-rail IB device names, and tests/bench_decode.py gained
# native API_KEY/VLLM_API_KEY auth (the hand-patched copy is no longer needed --
# /tmp/bench_leo.py is now just a BASE/MODEL sed of the stock harness).
# TWO CHANGES OF SUBSTANCE:
#  1. The hand-written --cudagraph-capture-sizes list was REMOVED from
#     EXTRA_ARGS in the kit's .env. start.sh:193-211 now derives the list from
#     GLM53_ADAPTIVE_K_SET + MAX_NUM_SEQS, but ONLY when the caller has not
#     supplied the flag -- ours had, so it was silently overriding the new
#     automatic list. Verified at boot: the derived list is
#     "1 2 3 4 5 6 8 9 10 12 15 16 20 24 32", byte-identical to the manual one,
#     and it now tracks the k-set on its own. (GLM53_ADAPTIVE_K itself still
#     defaults to off despite PR #169's title; the explicit ema still does it.)
#  2. GLM53_APC_RETENTION_INTERVAL_SWA exists now and is deliberately LEFT
#     EMPTY. It looks like the fix for the prefix-caching loss noted above as
#     the cost of leaving the fork. It is NOT: upstream's own qualification
#     measured, matched at 128K on a 2x Spark, edit-at-90% 112.49 s vs 14.70 s
#     and branch-at-90% 99.89 s vs 3.35 s, reusing ZERO tokens where the old
#     runtime reused 111,104. Edit-and-branch at depth is exactly this lane's
#     agent workload. Full reasoning is in the LEO 2026-09-13 block in the .env.
# Re-benched after the pull (same protocol, stock harness): structured
# 73.5 tok/s (accept 0.980), prose 32.4 (0.525) -- against 72.4 / 33.1 on
# 2026-09-11, i.e. unchanged within run-to-run noise, which is what a
# launcher-only range should do. KV pool still 883,552 / 1.04x at 850k.
#
# What changed by switching:
#   + No image build. MiaAI ships a prebuilt multi-arch image
#     ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3, so the fork's ~40 min
#     GPU-idle `docker build` step is gone (target removed below).
#   + Binds 0.0.0.0 out of the box, so no local start.sh edit is needed. The
#     port is protected by VLLM_API_KEY in the .env instead, with the matching
#     api_key on the LiteLLM glm-5.3-flash-exl3 entry — the fork bound loopback.
#   + Upstream HEAD is current (2026-09-10) vs the b5ab8091 the fork vendored.
#   - LOSES the fork's fine-grained prefix caching at 64-token grain, its
#     per-group KV retention and its long-prefill fairness cap. Those are the
#     reason the fork was picked in the first place (multi-turn agent TTFT
#     ~4 s -> ~1 s). MiaAI's own README says hits land only on 3584-token pages.
#     If multi-turn agent latency regresses, that is the first thing to suspect.
#   ~ Different serving shape: MNBT 7168 (not 3584), a MiaAI PROD value, and
#     850K ctx as of 2026-09-11. It did NOT fit at first (boot died in
#     _initialize_kv_caches: 13.46 GiB KV needed vs 11.8 available at gmu 0.85,
#     vLLM estimating a 616448 ceiling), so the lane ran at 600K for a day.
#     What made 850K fit is the FP8-dense opt-in, now on in the kit's .env:
#     GLM53_DENSE_FP8=dense,kda frees GPU memory and EXTRA_ARGS pins
#     --kv-cache-memory-bytes to 14 GiB so that memory becomes a pool big enough
#     for one 850K request (measured 883,552 tokens / 1.04x) instead of growing
#     into host RAM. GLM53_ADAPTIVE_K=ema is on with it (k chosen per request
#     from 2/4/7) and needs the --cudagraph-capture-sizes list in the same
#     EXTRA_ARGS. gmu stays 0.85. 900K — the value the 2026-09-07 CHANGELOG
#     shipped before upstream lowered it to 850K — is untested here.
#     Benched on this pair 2026-09-11 (temp 0, thinking off, 400 tok, median of
#     5, tests/bench_decode.py): structured 64.5 -> 72.4 tok/s (accept 0.938 ->
#     0.956), prose 27.8 -> 33.1 (0.341 -> 0.499). Both now beat upstream's
#     published 65.1 / 32.1.
#     TWO CAVEATS. (1) FP8 dense is upstream-PROVISIONAL and does move target
#     numerics (KL proxy 0.002-0.013 nats/position, no full KLD panel) — unlike
#     DFlash2 spec decode, which is bit-exact. (2) Host headroom is thinner than
#     upstream's: head MemAvailable ~2.9 GiB where their 14 GiB cap left ~5, and
#     earlyoom here runs -m 2 (~2.4 GiB) with --prefer vllm. That is the same
#     failure recorded under `flush` below. Watch it; the lever if it bites is
#     gmu (each 0.01 is ~1.2 GiB of host headroom), not the KV cap, which cannot
#     go below ~13.5 GiB without the boot refusing an 850K request.
#     ROLLBACK to the 600K/BF16 shape: .env.bak-600k-20260911 in the kit dir.
#     Weights are the SAME brandonmusic snapshot 1ae6d70
#     already in the head cache (MODEL_CACHE_NAME pins it so nothing re-downloads);
#     MiaAI mirror the bytes of 5ab363a8, five hours earlier the same day.
#     The drafter pin moves to dc77ff1, which this cache already holds.
#
# (Reederey87 fork clone ~/src/glm53-exl3 REMOVED 2026-09-22 at Leo's request;
# its site .env is archived in ~/src/.retired/. A rollback to that fork now means
# re-cloning github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark, restoring the
# .env from the archive, and re-adding the glm-exl3-build / glm-exl3-dry targets
# from git history.)
GLM_EXL3_DIR          := $(HOME)/src/glm53-exl3-miaai
# PARKED 2026-09-08 — superseded by `make qwen-flash` (tonyd2wild lane, below),
# which was verified on this pair the same day. The launch/dry/logs targets are
# commented out, not deleted: uncomment them (and `make qwen-flash` off) to roll
# back. The kit, its .env and the RadixArk checkpoint stay on disk.
# Qwen3.8-Flash-Next NVFP4 (125B-A3B hybrid MoE + 51B PLE + MTP head,
# multimodal) — MiaAI-Lab's vLLM TP2+EP+MTP3 kit, adopted 2026-09-06 @ c2325b2
# (github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks; clone + this pair's
# .env at ~/src/qwen38-flashnext-miaai-vllm — the .env header explains every
# local value). NOT a sparkrun recipe: the kit bind-mounts runtime patches
# (PLE FP8 resolver shim, MXFP8 kernel fallback, FP8-block MoE dispatch, MTP
# layer-index alias) that sparkrun cannot express, so it runs its own
# start.sh/stop.sh — docker per node over SSH, VLLM_HOST_IP + GLOO/NCCL/TP
# socket ifnames pinned per node to the CX7 link (the Wi-Fi control-plane
# problem `patch-sparkrun` exists for does not arise here). Same day-0 image
# vllm/vllm-openai:qwen38-flash-next (sha256 d464f3b4, identical on both nodes
# and identical to the one MiaAI measured on) and the RadixArk checkpoint both
# nodes already hold. MiaAI's numbers on 2x GB10, TP2+EP, MTP3, 262K: ~52 tok/s
# batch-1 (24.5 without MTP), 72.8% draft acceptance, ~2.9K tok/s prefill flat
# to 128K, x6 aggregate ~170 tok/s, ~11 min cold boot.
# Local deviations (all in the .env): gmu 0.80 (kit 0.835), 6 seqs (kit 8),
# bf16 KV (kit flipped its default to fp8 on 2026-09-05 — capacity we do not
# need at 262K x 6 and a quality trade on sparse attention), native 262K with
# YaRN off, port 8000. VERIFIED on this pair 2026-09-06, first launch (log:
# ~/bench/qwen38fn-miaai-20260906.launch.log): boot 13 min (weights 469 s +
# MTP drafter 75 s, engine init 141 s, graphs 4 s), 65.4 GiB weights/node,
# KV 29.77 GiB = 1,880,351 tokens = 7.17x at 262K, zero NV_ERR_NO_MEMORY, no
# MXFP8 fallback lines (RadixArk attention is BF16, so that patch is inert
# here), control plane on the CX7 link (VLLM_HOST_IP 10.100.200.2,
# *_SOCKET_IFNAME enp1s0f0np0). Through the LiteLLM proxy: thinking-on answer
# with reasoning_content, count-to-300 at 64.8 tok/s incl. prefill (ceiling
# prompt, thinking off), get_weather tool call parsed, red-square image ->
# "Red", 62K-token needle prompt answered correctly in 21.8 s (~2.9K tok/s
# prefill). MTP after those requests: 1134/1155 drafted tokens accepted, per
# position 382/378/374 of 385 drafts (decaying = drafter wired right; the
# count prompt inflates it, expect MiaAI's ~73% on real prompts). The boot log
# offers --kv-cache-memory=31601006183 (29.43 GiB) as the exact-fit pin.
# Supersedes tonyd2wild's SGLang lane (recipes/qwen3.8-flash-next-sglang.yaml,
# adopted 2026-09-03): it boots and then dies on the first >=2K-token prefill
# (mrope device-side assert in EAGLE draft-extend; ~/bench/qwen38fn-crash-
# 20260903.log), and tonyd2wild DELETED that lane upstream on 2026-09-05 —
# their repo is now vLLM-only too. Kept as `make qwen38fn-sglang` for
# reference. The getrefined hand launcher at ~/src/qwen38-flashnext-vllm is
# the older single-file version of the same vLLM stack (the MiaAI kit is
# based on it) and is no longer needed as a fallback.
# Upstream drops page cache on both nodes before launch (GB10 UMA) — `flush`.
QWEN38FN_DIR           := $(HOME)/src/qwen38-flashnext-miaai-vllm
QWEN38FN_SGLANG_RECIPE := recipes/qwen3.8-flash-next-sglang.yaml
# Qwen3.8-Flash-Next NVFP4 — tonyd2wild's vLLM TP2 "SPEED" lane, added 2026-09-08
# as `make qwen-flash` (github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark
# @ 6ad1c8f; clone at ~/src/qwen38-flashnext-tony — the same clone the parked
# SGLang lane came from, pulled forward from b515104; that SGLang lane now lives
# under lanes/sglang-tp2/ upstream, unchanged). Upstream rebuilt the repo on vLLM
# 2026-09-05: nightly vllm/vllm-openai:nightly-8a728663 (vLLM main 2026-09-04)
# + five bind-mounted overlays (PR #55375 PLE conv-state stride fix, PR #54846
# x3 fp8 KV on the QSA path, their modelopt.py MTP-loading fixes; provenance +
# sha256 in the patch dir's PROVENANCE.md) on the OFFICIAL
# nvidia/Qwen3.8-Flash-Next-NVFP4 checkpoint — NOT the RadixArk build the MiaAI
# kit uses (different config, 23 KB vs 504-byte hf_quant_config, 10 shards vs
# 419 per-layer expert files). 133 GB at
# /var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia on BOTH nodes (head pulled it
# over Wi-Fi with `uvx hf download`, worker rsynced over CX7).
# config.json in that dir is PINNED to HF revision fab0aecb (2026-09-03, the
# snapshot tonyd2wild, sfxnz and MiaAI all validated). NVIDIA's fc694b54
# (2026-09-05 23:36 UTC, "Fix MTP serving metadata and instructions") changed
# exactly one line, the MTP experts' quant_algo FP8_BLOCK_SCALES -> FP8_PB_WO
# (weights and hf_quant_config.json are byte-identical between the two), and
# this nightly's mixed-precision MoE dispatch + tonyd2wild's overlay do not
# route FP8_PB_WO: the draft head loads unquantized and boot dies at shard
# 11/11 with "mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'"
# (first boot here, 2026-09-08 16:16, both ranks). NVIDIA's copy is kept next
# to it as config.json.fc694b54; a fresh `hf download` would undo the pin, and
# `qwen-flash-dry` / CHECK=1 fail loudly on FP8_PB_WO. Worth an upstream note
# to tonyd2wild (one-token overlay change: accept FP8_PB_WO in the MoE branch).
# SPEED profile = n-gram table in unified memory (stock loader), decode CUDA
# graphs with torch.compile OFF ({"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}
# — Inductor duplicates the 47.7 GB table during compile and rebooted their
# Sparks), MTP3, 6 seqs, --max-num-batched-tokens 4096 (their biggest single
# lever, +50% under compile-off; NEVER pair it with compile on: 8-15 tok/s),
# fp8_e4m3 KV, gmu 0.70, native 262K, prefix caching off, FlashInfer autotune
# off, VLLM_USE_DEEP_GEMM=0. Upstream measured on 2x GB10: 53.7 tok/s median
# single stream over 40 real prompts, 97.9 aggregate at x6, 180 ms TTFT, KV
# pool 1.97M tokens (7 x 262K), ~2.8K tok/s prefill at 28K. PROFILE=context
# gives their CONTEXT profile (table on disk via their patch, compile on, MTP4,
# 8 seqs, gmu 0.80): 5.87M-token pool at 35.8 tok/s.
# NOT a sparkrun recipe (same reason as the MiaAI kit: bind-mounted overlays).
# The launcher is upstream's launch/qwen38fn-nvidia-tp2.sh with this pair's
# deviations marked LEO: in tools/qwen-flash-tp2.sh: LANE=leo (10.100.200.2
# head / .1 worker, cross-wired CX7 = head f0 / worker f1, HCA + socket ifnames
# derived per rank from the host IP), THINKING knob (upstream ships OFF; the
# two targets below set it), PROFILE + CHECK conveniences, nothing else.
# Container vllm_qwen38fn (upstream's name) on both nodes, served name
# qwen3.8-flash-next on :8000 — LiteLLM/opencode/Zed entries unchanged.
# Upstream's run order: worker (rank 1) first, then head; the targets do that.
# One model at a time: `make stop` first (the launcher does not check for a
# busy GPU).
QWEN_FLASH_DIR      := $(HOME)/src/qwen38-flashnext-tony
QWEN_FLASH_LAUNCHER := tools/qwen-flash-tp2.sh
QWEN_FLASH_PATCHES  := $(HOME)/patches/qwen4exp-ple-mmap
# Knobs for `make qwen-flash*` (command line): PROFILE=speed|context,
# KV_DTYPE=fp8_e4m3|auto (auto = bf16, the MiaAI-lane choice; halves the pool),
# DRAFT_VOCAB=65536 (reduced-vocab MTP draft: prose +10%, short structured -2..3 tok/s), GMU=.
PROFILE     ?= speed
KV_DTYPE    ?=
DRAFT_VOCAB ?=
GMU         ?=
QWEN_FLASH_ENV := PROFILE=$(PROFILE)
ifneq ($(strip $(KV_DTYPE)),)
QWEN_FLASH_ENV += KV_DTYPE=$(KV_DTYPE)
endif
ifneq ($(strip $(DRAFT_VOCAB)),)
QWEN_FLASH_ENV += DRAFT_VOCAB=$(DRAFT_VOCAB)
endif
ifneq ($(strip $(GMU)),)
QWEN_FLASH_ENV += GMU=$(GMU)
endif

# MiMo-V2.6-Flash-RL (Xiaomi; 309B total / 15B active, MXFP4 experts, fp8 attention,
# 1M native ctx, text + image + video + audio in) — MiaAI-Lab's SGLang TP2/EP2 kit,
# `make mimo` SINCE 2026-09-22 (github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks
# @ 201be3e; clone at ~/src/mimo26-sglang-miaai on branch leo/worker-socket-ifname,
# ONE local commit over upstream main: WORKER_GLOO_SOCKET_IFNAME /
# WORKER_NCCL_SOCKET_IFNAME in start.sh, because this pair is cross-wired (head
# enp1s0f1np1 <-> worker enp1s0f0np0) and the kit passes one socket ifname to both
# ranks — gloo refuses a comma list naming a port the node lacks, tested. Update
# with `git fetch && git rebase origin/main`, not --ff-only. Site profile = its
# .env, every local value marked LEO; the kit reads ONLY that file, so knobs are
# edited there, not passed on the make line.)
# NOT a sparkrun recipe and unlike the sibling MiaAI kits it REQUIRES NFS: the
# worker mounts the head's /var/tmp/models/MiMo-V2.6-Flash-RL over the CX7 link
# (kit-built mimo26-nfs container on the head, docker nfs volume on the worker);
# the worker's own copy of the weights is the rollback lane's and unused here.
# Engine: SGLang nightly-dev-cu13-20260921 (first image keeping the experts packed
# MXFP4 — anything older expands them to FP8, ~151 GiB/rank, and takes the node
# down; boot.py refuses such an engine) built into mimo26-spark:local with ffmpeg +
# torchcodec, then docker save | ssh docker load to the worker; a BASE_IMAGE change
# in .env rebuilds. patches/sitecustomize.py (bind-mounted, PYTHONPATH-first) clones
# one tensor at a time on the way to the GPU (96 s/rank vs 525 s stock mmap) and
# reshards the MTP draft's fused qkv. First launch: doctor -> build (~5-10 min) ->
# share -> serve; boot ~2-3 min after that. Head :8000 (kit 8888), served name
# mimo-v2.6-flash, parsers `mimo`. Drafter DFLASH k=8 with decode CUDA graphs
# (code 69.8 / prose 25.8 tok/s upstream at 1 stream, 190/103 aggregate at 8;
# EAGLE MTP is the kit default: prose 35.0 / code 46.0, no graphs, 1.73x the KV
# pool — one .env edit + `./start.sh restart` to switch, README "Pick a drafter").
# KV pool with DFlash 1,664,704 tokens = 1.59x a full 1M context; prefill 2,558
# tok/s at 8K, 764 at 256K. THINKING IS ON BY DEFAULT on this engine (vendor
# template; no server-side default kwargs in SGLang) — LiteLLM pins
# enable_thinking=false for flag-less requests so the proxy behaves as before;
# a direct backend call without chat_template_kwargs thinks. No repetition
# penalty (the tonyd2wild lane had 1.05 for agent tool-call loops; upstream
# measured it flat on DFlash and removed it).
# VERIFIED 2026-09-22 evening, third boot: at the kit's MEM_FRACTION_STATIC 0.93
# and at 0.91 the 4 GiB MemAvailable guard killed the engine on BOTH ranks right
# after the KV pool — FlashInfer's MoE autotune runs there and costs ~8.5 GB for
# ~73 s (sampled minimum 5.5 GiB head / 6.7 GiB worker at 0.88). This pair runs
# 0.88 (LEO in .env) and keeps the guard at 4; pool 1,224,832 tokens = 1.17x a
# full 1M request. Boot ~7 min once the image exists (weights 96 s/rank on both
# NVMe and NFS, autotune 73 s, capture 17 s). Smoke 42; proxy: no level = thinking
# off, high = reasoning_content, tool calls parse. Decode (idle, temp 0, off,
# median of 3): structured 82.7 / code 56.5 / prose 20.7 tok/s with DFlash
# accepting 7.7 / 6.9 / 5.7 of 8 — a few % under the vLLM lane at one stream;
# prefill 1,448 tok/s on a 97K needle (answered exactly), +24% over vLLM.
MIMO26_DIR        := $(HOME)/src/mimo26-sglang-miaai

# ROLLBACK lane `make mimo-vllm` (was `make mimo` 2026-09-22 until the SGLang kit
# above took the name the same day): tonyd2wild's vLLM TP2 + DFlash k=7 kit.
# (github.com/tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark @ 18705d6 — cloned at
# 7dce2a5, pulled to a05d97d (docs-only rebench) then 18705d6 (sampling defaults: --generation-config auto + repetition_penalty 1.05, REP_PENALTY in mimo.env; fixes agent tool-call loops), all 2026-09-22; clone + this
# pair's launch/mimo.env at ~/src/mimo26-flash-tony — every local value is
# marked LEO: in that file). NOT a sparkrun recipe: the kit bind-mounts four
# patched vLLM files over the image's copies (fused fp8 QKV loader for the
# TP4-presharded checkpoint, SupportsEagle3 on the Omni wrapper so DFlash
# works, fp8 KV that actually applies on the DiffKV backend, opt-in drafter
# value scale DFLASH_VSCALE=1 — measured no gain upstream) plus a corrected
# dflash/config.json (the release has a trailing comma) and soundfile/PyAV
# for audio input, all staged in /var/tmp/mimo-cache by the kit's setup.sh.
# Image ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 — the same
# public tag the parked glm53-dflash2 lane uses; its layers were already on
# both nodes, so the pull was a re-tag. Weights 166 GiB (65 shards + the
# DFlash drafter) at /var/tmp/models/MiMo-V2.6-Flash-RL on BOTH nodes: head
# over Wi-Fi with `hf download`, worker rsync'd over CX7 by
# `make mimo-sync-weights` (no NFS between this pair).
# Upstream on 2x GB10, fp8 KV, 300K ctx, gmu 0.90, 8 seqs, DFlash 7, marlin
# MoE, DeepGEMM off: 53.3 tok/s single stream (code 70.5, prose 25.9), 155.8
# aggregate at x6, TTFT 0.37 s, prefill 1,947 tok/s at 2K down to 656 at 250K,
# KV pool 1.87M tokens (six full 300K requests at once). 1M ctx is UNTESTED
# upstream (a 1M request needs ~3.3x the per-request blocks); `make mimo
# MAX_MODEL_LEN=1000000` (or MAXLEN in launch/mimo.env) raises it; knobs above RUN.
# Local deviations (all in launch/mimo.env): HEAD_IP 10.100.200.2, ADDR_RANGE
# 10.100.200.0/24, PORT 8000 (kit: 8888) so LiteLLM and the dashboard find it
# where every other lane serves. The worker's NIC/HCA (cross-wired pair: head
# f0 / worker f1) go on its command line below — serve.sh lets the calling
# environment win over the file. Served name mimo-v2.6-flash; thinking OFF
# server-side (kit default, THINKING=false — with it on, reasoning can leak
# into content for clients that do not read reasoning_content); a request
# turns it on with chat_template_kwargs enable_thinking=true. Reasoning and
# tool-call parsers are both `mimo`. Vendor sampling: temp 1.0, top_p 0.95.
# Upstream's run order: worker (rank 1) first, then head; the target does that.
# The `|| [ $$? -eq 1 ]` on the worker lines: serve.sh ends with a head-only
# `[ "$$R" = 0 ] && echo`, so rank 1 exits 1 after a SUCCESSFUL docker run and
# make would otherwise abort before the head starts. Real failures exit 2-4
# (env/model/patches) or 125+ (docker) and still stop the target.
# One model at a time: `make stop` first — serve.sh waits up to 150 s for
# MemAvailable to reach GMU x 121.69 GiB and then runs docker anyway.
# VERIFIED 2026-09-22, first launch: boot 14 min (head 663 s of shard reads at
# ~10 s/shard from NVMe; the worker, copy still in page cache, 159 s), 83.0 GiB
# weights/node, KV pool 1,950,120 tokens = 6.50x at 300K (13.14 GiB; boot log
# offers --kv-cache-memory 13275789722 as the exact-fit pin), MemAvailable
# 9.0/10.3 GiB head/worker after boot, 7.8/9.7 after a 76K prompt. Thinking
# off by default (empty reasoning), enable_thinking=true -> reasoning in the
# `reasoning` field, content clean; tools parse both ways; the kit's vision
# test reads all four elements; 76,497-token needle answered in 65 s.
MIMO26_VLLM_DIR        := $(HOME)/src/mimo26-flash-tony
MIMO26_VLLM_MODEL      := /var/tmp/models/MiMo-V2.6-Flash-RL
MIMO26_VLLM_CACHE      := /var/tmp/mimo-cache
MIMO26_VLLM_WORKER_ENV := IFACE=enp1s0f0np0 HCA=rocep1s0f0

# Optional overrides — set on the command line, e.g.
# make deepseek MAX_MODEL_LEN=1000000 GPU_MEM=0.85
MAX_MODEL_LEN ?=
GPU_MEM       ?=

# Assemble override flags only when the corresponding var is set.
OVERRIDES :=
ifneq ($(strip $(MAX_MODEL_LEN)),)
OVERRIDES += --max-model-len $(MAX_MODEL_LEN)
endif
ifneq ($(strip $(GPU_MEM)),)
OVERRIDES += --gpu-mem $(GPU_MEM)
endif

# Knobs for `make mimo-vllm*` (command line; the SGLang `make mimo` lane reads its
# .env only), forwarded to BOTH ranks — serve.sh lets
# the calling environment win over launch/mimo.env: MAX_MODEL_LEN=1000000 (kit
# tested 300000; 1M is untested upstream), GMU=, KV_DTYPE=fp8|auto (auto = bf16
# KV, halves the pool), SEQS=, MIMO_THINKING=true|false (server default; kit false).
SEQS          ?=
MIMO_THINKING ?=
MIMO26_VLLM_ENV :=
ifneq ($(strip $(MAX_MODEL_LEN)),)
MIMO26_VLLM_ENV += MAXLEN=$(MAX_MODEL_LEN)
endif
ifneq ($(strip $(GMU)),)
MIMO26_VLLM_ENV += GMU=$(GMU)
endif
ifneq ($(strip $(KV_DTYPE)),)
MIMO26_VLLM_ENV += KV_DTYPE=$(KV_DTYPE)
endif
ifneq ($(strip $(SEQS)),)
MIMO26_VLLM_ENV += SEQS=$(SEQS)
endif
ifneq ($(strip $(MIMO_THINKING)),)
MIMO26_VLLM_ENV += THINKING=$(MIMO_THINKING)
endif

RUN := $(SPARKRUN) run --cluster $(CLUSTER)

# The worker node (node_1), addressed over the cluster link the way sparkrun does.
WORKER ?= 10.100.200.1

# 3-node lanes (2026-09-29): the Gigabyte AI TOP ATOM (aitopatom-4b9d, GB10) is rank 2.
# CX7 directed ring, each node's Port0 -> the next node's Port1:
#   spark-f31f P0 -> atom P1   10.100.210.0/24 (+ .211 on the second PCIe function)
#   atom P0 -> spark-d306 P1   10.100.220.0/24 (+ .221)
#   spark-d306 P0 -> f31f P1   10.100.200.0/24 (+ .201) = the 2-node link, same IPs as before
# The netplan (MTU 9000, plus host routes so every fabric IP answers from every node) is
# /home/leo/cx7-ring.sh on each node. flush/cache-flusher take WORKERS; the 3-node
# targets pass both workers.
WORKER2 ?= 10.100.210.3
WORKERS ?= $(WORKER)
WORKERS3 := $(WORKER) $(WORKER2)

.PHONY: help deepseek deepseek-sparkrun ds41 ds41-status logs-ds41 stop-ds41 glm-exl3 glm-exl3-ablit glm-exl3-ablit-fetch glm-exl3-ablit-check qwen38fn qwen38fn-sglang qwen-flash qwen-flash-no-thinking qwen-flash-sync \
        mimo mimo-vllm mimo-vllm-sync mimo-vllm-sync-weights \
        deepseek-dry qwen38fn-dry qwen-flash-dry mimo-dry mimo-vllm-dry \
        stop stop-deepseek stop-glm-exl3 stop-qwen38fn stop-qwen38fn-sglang stop-qwen-flash stop-mimo stop-mimo-vllm \
        status logs logs-glm-exl3 logs-qwen-flash logs-mimo logs-mimo-vllm list flush patch-sparkrun cache-flusher stop-cache-flusher \
        glm-exl3-tp3 glm-exl3-tp3-status stop-glm-exl3-tp3 logs-glm-exl3-tp3 \
        deepseek-tp3 deepseek-tp3-prepare deepseek-tp3-env stop-deepseek-tp3 flush3 cache-flusher3 free-nfs \
        jspark3 jspark3-verify jspark3-status logs-jspark3 stop-jspark3 \
        ds41x3 ds41x3-build ds41x3-pack ds41x3-status logs-ds41x3 stop-ds41x3 \
        glm-tf glm-tf-status logs-glm-tf stop-glm-tf
# (qwen38fn, qwen38fn-dry, logs-qwen38fn left out on purpose: parked 2026-09-08, see the MiaAI block)

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- launch ---------------------------------------------------------------

deepseek: flush cache-flusher ## Launch DeepSeek-V4-Flash-Vision-Exp + DSpark MTP=6 (MiaAI-Lab kit, 2-node, NVFP4 KV, 1M ctx)
	cd $(DEEPSEEK_MIAAI_DIR) && ./start-deepseek-v4-flash-dspark.sh

deepseek-sparkrun: cache-flusher ## ROLLBACK: the tonyd2wild sparkrun recipe (DSpark k=5, 12 seqs, gmu 0.85)
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES)

ds41: flush cache-flusher ## Launch DeepSeek-V4.1-Flash EXL3 2.9bpw (MiaAI kit, 2-node, 600K ctx, text-only) -- needs ./download.sh first
	cd $(DS41_DIR) && ./start.sh start

ds41-status: ## Status of the DeepSeek-V4.1 EXL3 lane
	cd $(DS41_DIR) && ./start.sh status

logs-ds41: ## Tail the DeepSeek-V4.1 EXL3 head container
	cd $(DS41_DIR) && ./start.sh logs

# `flush` first, since 2026-09-05: NVRM refused part of the 6 GiB KV carve-out at
# boot (dmesg: NV_ERR_NO_MEMORY from _memdescAllocInternal @ mem_desc.c:1359, one
# second before vLLM logged "reserved 6.0 GiB memory for KV Cache") on nodes whose
# page cache had not been dropped. (vLLM's own "Available RAM: ~20 GiB" line at
# weight-load start is NOT evidence of that — it prints the same figure flushed or
# not; read /proc/meminfo instead.) vLLM does not notice: it served short prompts for 4.5 h, then the
# first ~25K-context decode stalled (`RPC call to sample_tokens timed out`, no
# traceback on either rank, worker rank 3 GB into swap). Upstream's KV-hunt record
# calls this the phantom reserve; its rule is drop_caches on every rank right
# before launch. After a launch, `sudo dmesg -T | grep NV_ERR_NO_MEMORY` must be
# clean at the "reserved" second and a >=28K-token prompt must pass — a short
# prompt cannot see this. If NVRM still refuses, the next lever is the pin itself:
# kv_cache_memory 5905580032 (5.5 GiB, upstream's stable TP2 record) in the yaml.
#
# `patch-sparkrun` too, since 2026-09-05: the relaunch after that stall failed in
# gloo's first barrier, and the reason turned out to be the control plane, not
# memory. sparkrun puts the torch master address AND every host's GLOO/NCCL/TP
# socket pin on the default-route interface — here wlP9s9, because the 10GbE
# ports have no carrier — and the Optus mesh black-holes Spark<->Spark Wi-Fi
# after a roam (ARP for the peer resolves to the satellite's MAC; 100% loss on
# 192.168.0.x while 10.100.200.x is clean). vLLM's own get_ip() follows the
# default route too (the engine core's mq_connect_ip: scheduler broadcast and
# worker responses), so moving only master-addr/gloo left a launch hanging
# silently after "reserved 6.0 GiB". tools/patch_sparkrun_wifi.py makes
# sparkrun's two detect scripts substitute the up RDMA netdev for a wireless
# default interface and its env builder emit VLLM_HOST_IP per host, so the whole
# control plane rides the CX7 link like NCCL already does. Verify after a launch:
#   docker inspect <node_0> | grep -E "VLLM_HOST_IP|GLOO_SOCKET_IFNAME" -> 10.100.200.2 / enp1s0f1np1
#   grep mq_connect_ip /tmp/sparkrun_serve.log (in-container)         -> 10.100.200.2, not 192.168.0.120
# Cabling the 10GbE ports would make the patch redundant (wired default route).
glm-exl3: flush cache-flusher ## Launch GLM-5.3-Flash EXL3 4bpw + DFlash2 k=7 (MiaAI-Lab kit, 2-node, 850K ctx, FP8 dense + adaptive-k) — the GLM 5.3 lane
	cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh start

# ABLIT variant of the same lane (added 2026-09-30): MiaAI's runtime abliteration
# (README "Abliteration (ABLIT=1)") — o_proj L15-45 byte-transplanted at load from
# the dealign donor (L0-14 stock anchors; the DFlash2 drafter is never touched).
# Same TR3 weights, image and .env as glm-exl3, nothing rewritten on disk. The FP8
# dense pass (GLM53_DENSE_FP8=all covers o_proj) runs in process_weights_after_loading,
# AFTER the load_weights hook, so the edit lands on BF16 and is then quantised.
# .env is sourced with set -a, so ABLIT and the served name MUST ride on the
# start.sh command line (a caller export beats .env). ABLIT_METHOD=transplant, not
# auto: auto silently falls back to proj without ablit/transplant/, which garbles
# sampled output. Stop with stop-glm-exl3 (same containers).
glm-exl3-ablit-fetch: ## One-time: fetch the ablit o_proj transplant (~2.7 GiB of HF range reads) into the GLM kit's ablit/transplant/ — not while a 3-node lane serves
	cd $(GLM_EXL3_DIR) && python3 ablit/fetch_transplant.py

glm-exl3-ablit-check:
	@test -f $(GLM_EXL3_DIR)/ablit/transplant/MANIFEST.json || { echo "$(GLM_EXL3_DIR)/ablit/transplant/ is missing: run make glm-exl3-ablit-fetch first" >&2; exit 1; }

glm-exl3-ablit: glm-exl3-ablit-check flush cache-flusher ## Launch GLM-5.3-Flash EXL3 ABLITERATED (MiaAI ABLIT=1 transplant, o_proj L15-45; 2-node, 850K ctx; served glm-5.3-flash-exl3-ablit)
	cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ABLIT=1 ABLIT_METHOD=transplant SERVED_MODEL_NAME=glm-5.3-flash-exl3-ablit ./start.sh start

# UN-PARKED 2026-09-15 at Leo's request: this is the qwen-flash lane now.
# Reconfigured off its old defaults — nvidia NVFP4 checkpoint (RadixArk is NOT
# on either Spark; that .env would have meant a ~133 GB download), fp8 KV, and
# YaRN to 1M. Measured at boot 2026-09-15: KV pool 4,185,628 tokens = 4.19x
# concurrency at a full 1M request, vs MAX_NUM_SEQS=3. `make qwen-flash`
# (tonyd2wild) is still here and is still the faster lane at 262K.
qwen38fn: flush ## Launch Qwen3.8-Flash-Next NVFP4 (MiaAI kit, 2-node, nvidia ckpt, fp8 KV, MTP3, YaRN 1M ctx)
	cd $(QWEN38FN_DIR) && ./start.sh --launch

qwen38fn-sglang: ## PARKED — tonyd2wild SGLang lane (dies on first real prefill); reference only
	$(RUN) $(QWEN38FN_SGLANG_RECIPE) $(OVERRIDES)

qwen-flash: flush cache-flusher qwen-flash-sync ## Launch Qwen3.8-Flash-Next NVFP4 (nvidia ckpt) — tonyd2wild vLLM TP2 SPEED lane, 262K ctx, fp8 KV, MTP3, thinking ON
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'THINKING=1 $(QWEN_FLASH_ENV) bash ~/qwen-flash-tp2.sh 1'
	THINKING=1 $(QWEN_FLASH_ENV) bash $(QWEN_FLASH_LAUNCHER) 0
	@echo "booting (~10 min to serve): make logs-qwen-flash — ready at 'Application startup complete'"

qwen-flash-no-thinking: flush cache-flusher qwen-flash-sync ## Same lane with enable_thinking false server-side (clients can still send enable_thinking per request)
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'THINKING=0 $(QWEN_FLASH_ENV) bash ~/qwen-flash-tp2.sh 1'
	THINKING=0 $(QWEN_FLASH_ENV) bash $(QWEN_FLASH_LAUNCHER) 0
	@echo "booting (~10 min to serve): make logs-qwen-flash — ready at 'Application startup complete'"

qwen-flash-sync: ## Ship upstream's patch dir + the launcher to the worker (idempotent; rerun after a git pull in $(QWEN_FLASH_DIR))
	mkdir -p $(QWEN_FLASH_PATCHES)
	rsync -a --delete $(QWEN_FLASH_DIR)/single-spark-vllm-tp1/patch/ $(QWEN_FLASH_PATCHES)/
	rsync -a --delete $(QWEN_FLASH_PATCHES)/ $(WORKER):patches/qwen4exp-ple-mmap/
	rsync -a $(QWEN_FLASH_LAUNCHER) $(WORKER):qwen-flash-tp2.sh

mimo: flush cache-flusher ## Launch MiMo-V2.6-Flash-RL (MiaAI-Lab SGLang kit, 2-node TP2/EP2, MXFP4 experts, fp8 KV, 1M ctx, DFlash k=8, image+video+audio) — doctor, build if needed, NFS share, serve
	cd $(MIMO26_DIR) && ./start.sh serve
	@echo "serving: make logs-mimo — curl -s localhost:8000/v1/models lists mimo-v2.6-flash (thinking ON by default at the backend; LiteLLM pins it off)"

mimo-vllm: flush cache-flusher mimo-vllm-sync ## ROLLBACK: launch MiMo-V2.6-Flash-RL on tonyd2wild's vLLM TP2 + DFlash k=7 kit (fp8 KV, 300K ctx, thinking OFF server-side)
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'cd ~/src/mimo26-flash-tony && $(MIMO26_VLLM_ENV) $(MIMO26_VLLM_WORKER_ENV) bash launch/serve.sh 1 || [ $$? -eq 1 ]'
	cd $(MIMO26_VLLM_DIR) && $(MIMO26_VLLM_ENV) bash launch/serve.sh 0
	@echo "booting (~11 min to serve): make logs-mimo-vllm — ready when curl -s localhost:8000/v1/models lists mimo-v2.6-flash"

mimo-vllm-sync: ## Ship the vLLM kit (launch/serve.sh + mimo.env, patches) and the staged patch files + audio libs to the worker (idempotent; rerun after a git pull or setup.sh)
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'mkdir -p ~/src/mimo26-flash-tony $(MIMO26_VLLM_CACHE)'
	rsync -a --delete --exclude .git $(MIMO26_VLLM_DIR)/ $(WORKER):src/mimo26-flash-tony/
	rsync -a $(MIMO26_VLLM_CACHE)/mimo_v2.py $(MIMO26_VLLM_CACHE)/mimo_v2_omni.py $(MIMO26_VLLM_CACHE)/triton_attn_diffkv.py $(MIMO26_VLLM_CACHE)/qwen3_dflash.py $(MIMO26_VLLM_CACHE)/dflash-config.fixed.json $(MIMO26_VLLM_CACHE)/pyextra $(WORKER):$(MIMO26_VLLM_CACHE)/

mimo-vllm-sync-weights: ## Copy the 166 GiB MiMo checkpoint to the worker over the CX7 link for the vLLM lane (once; a rerun only checks; the SGLang lane mounts the head's copy over NFS instead)
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'mkdir -p $(MIMO26_VLLM_MODEL)'
	rsync -a --exclude .cache --info=progress2 $(MIMO26_VLLM_MODEL)/ $(WORKER):$(MIMO26_VLLM_MODEL)/

## --- 3-node lanes: spark-f31f + spark-d306 + aitopatom-4b9d ---------------
#
# One workload at a time across the whole ring: stop the 2-node lane first.
# Only two kits have a TP=3 path (checked 2026-09-29): MiaAI's GLM EXL3
# (start-tp3.sh) and MiaAI's DS4 DSpark (start-tp3.sh). Qwen Flash-Next, MiMo
# and DS4.1 EXL3 are 2-node kits, and every tonyd2wild kit is TP2/TP4 — their
# head/KV/expert counts do not divide by 3. Both lanes serve on :8000 under
# their own name (…-tp3), so the dashboard and LiteLLM tell them apart.

# MiMo's exporter (mimo26-nfs) holds the head's kernel nfsd, and GLM's
# files/nfs-share.sh only adopts glm53-nfs / vllm-fn-nfs / glm53fp8-nfs /
# dsv41-nfs, so it would start a second nfsd and fail. Drop MiMo's while MiMo
# is not serving; stop-glm-exl3-tp3 drops glm53-nfs again, so `make mimo`
# recreates its own exporter exactly as before.
free-nfs:
	@if docker ps --format '{{.Names}}' | grep -qx 'mimo26-tp2-head'; then echo "MiMo is serving: make stop-mimo first" >&2; exit 1; fi
	@docker rm -f mimo26-nfs >/dev/null 2>&1 && echo "removed mimo26-nfs (make mimo recreates it)" || true

# .env.tp3 in the kit (LEO 2026-09-29 comments): ranks 10.100.200.2 / .200.1 /
# .210.3, dual-port CX7 pins on every rank, bootstrap on enP7s7, NFS_SHARE=1
# (one 164 GiB copy; also dodges MiaAI #287), 40 GiB KV pin, 1M ctx, :8000.
glm-exl3-tp3: flush3 cache-flusher3 free-nfs ## Launch GLM-5.3-Flash EXL3 on THREE nodes (MiaAI start-tp3.sh: TP=3 + EP, 1M ctx, served glm-5.3-flash-exl3-tp3)
	cd $(GLM_EXL3_DIR) && ./start-tp3.sh start

glm-exl3-tp3-status: ## Status of the 3-node GLM lane
	cd $(GLM_EXL3_DIR) && ./start-tp3.sh status

logs-glm-exl3-tp3: ## Tail the 3-node GLM head container
	cd $(GLM_EXL3_DIR) && ./start-tp3.sh logs

stop-glm-exl3-tp3: ## Stop the 3-node GLM lane (all three ranks) and its NFS exporter
	-cd $(GLM_EXL3_DIR) && ./start-tp3.sh stop
	-docker rm -f glm53-nfs 2>/dev/null

# DS4 reads ONE env file (ENV_FILE overrides it). The 3-node lane runs from a
# generated .env.dspark.tp3 = .env.dspark with tools/ds4-tp3.env's keys
# replaced, so .env.dspark stays the single source of truth.
DS4_TP3_ENV := $(DEEPSEEK_MIAAI_DIR)/.env.dspark.tp3

deepseek-tp3-env:
	@python3 tools/env-overlay.py $(DEEPSEEK_MIAAI_DIR)/.env.dspark tools/ds4-tp3.env $(DS4_TP3_ENV)

deepseek-tp3-prepare: deepseek-tp3-env ## One-time for 3-node DeepSeek: verify/fill the checkpoint + image on both workers (DSPARK_WORKER_HF_NFS=0)
	cd $(DEEPSEEK_MIAAI_DIR) && ENV_FILE=$(DS4_TP3_ENV) ./prepare-dspark-model-cache.sh --yes

deepseek-tp3: flush3 cache-flusher3 deepseek-tp3-env ## Launch DeepSeek-V4-Flash-Vision-Exp + DSpark on THREE nodes (MiaAI start-tp3.sh, 8->9 group pad, served deepseek-v4-flash-vision-exp-tp3)
	cd $(DEEPSEEK_MIAAI_DIR) && ENV_FILE=$(DS4_TP3_ENV) ./start-tp3.sh

stop-deepseek-tp3: deepseek-tp3-env ## Stop the 3-node DeepSeek lane (all three ranks)
	-cd $(DEEPSEEK_MIAAI_DIR) && ENV_FILE=$(DS4_TP3_ENV) ./stop-deepseek-v4-flash-dspark.sh

# JSpark3 v1.8.4 (github.com/jakejharris/jspark3 @ 64220b0, Apache-2.0 + AGPL parts; the
# DFlash2 draft it always loads is CC BY-NC-ND). GLM-5.3-Flash stock weights, TP3 + EP on
# all three GB10s, its own image built on the head. Source + prepared runtime (operator.env,
# receipts) live in ~/jspark3/src on f31f; each rank has ~/jspark3/{recipe-v1.8.4,models,
# sources,work}. LOCAL PATCH: the prepared recipe's remote_preflight.py also accepts DMI
# "AI TOP ATOM" (the Atom, same P4242 board) and its SHA256SUMS were re-sealed; the upstream
# originals sit beside it as *.upstream-v1.8.4. Fixed by the recipe: API 0.0.0.0:8888 with
# NO auth (jspark3-api-guard.service on f31f limits who can reach it), headless hosts
# (multi-user.target, nvidia-drm modeset=1 fbdev=0), served name glm-5.3-flash.
# A restart is always stop --remove + fresh preflight + start (~13 min).
JSPARK3_RUNTIME := $(HOME)/jspark3/src/jspark3-runtime-v1.8.4
JSPARK3_FLEET   := cd $(JSPARK3_RUNTIME)/recipe && python3 -B scripts/fleetctl.py

jspark3: flush3 ## Launch JSpark3 v1.8.4 on THREE nodes (GLM-5.3-Flash TP3, :8888, served glm-5.3-flash; preflight ~5 min + start ~8 min)
	@if [ -e $(JSPARK3_RUNTIME)/service.json ]; then echo "JSpark3 is running (service.json exists): make stop-jspark3 first" >&2; exit 1; fi
	$(JSPARK3_FLEET) preflight --env-file ../operator.env --output ../preflight.json
	$(JSPARK3_FLEET) start --env-file ../operator.env --preflight ../preflight.json --preflight-sha256 $$(sha256sum ../preflight.json | cut -d' ' -f1) --manifest ../service.json --confirm START-JSPARK3

jspark3-verify: ## JSpark3's own end-to-end verify (health, correctness, >32K retrieval, memory/no-swap)
	$(JSPARK3_FLEET) verify --env-file ../operator.env --manifest ../service.json --output ../verify.json --log-output ../verify-rank0.log

jspark3-status: ## Status of the JSpark3 fleet
	$(JSPARK3_FLEET) status --env-file ../operator.env --manifest ../service.json

logs-jspark3: ## Tail the JSpark3 rank-0 container
	docker logs -f --tail 200 jspark3-v16-rank0

stop-jspark3: ## Stop AND remove JSpark3's three rank containers (a restart needs a fresh preflight anyway)
	-$(JSPARK3_FLEET) stop --env-file ../operator.env --manifest ../service.json --confirm STOP-JSPARK3 --remove --remove-confirm REMOVE-JSPARK3 && mv $(JSPARK3_RUNTIME)/service.json $(JSPARK3_RUNTIME)/service-stopped-$$(date +%Y%m%dT%H%M%S).json
	tools/jspark3-archive-evidence.sh $(WORKER) $(WORKER2)

# GLM-5.3-Flash on TensorFold: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ ed026ef
# (Apache-2.0; TensorFold v0.5.0 + 52 patches; the DFlash2 draft it loads is CC BY-NC-ND), cloned
# 2026-10-01 as ~/src/glm53-tensorfold-miaai. TWO nodes only (TensorFold runs GLM on exactly two
# ranks): head f31f + worker d306 over 10.100.200.0/24. scripts/local.sh points it at the
# brandonmusic TR3 4bpw snapshot both nodes already hold (the kit default is MiaAI byte-identical
# mirror) and sets PREPARE=0; the prebuilt GHCR image (v0.5.0-cefe8bf45d07) was pulled by hand on
# both nodes and tagged tensorfold-glm53:v0.5.0. API 0.0.0.0:8888 with NO auth, the same port as
# JSpark3, so jspark3-api-guard covers it. Served GLM-5.3-Flash-EXL3, 1M ctx, 4 requests at once,
# FP8 KV, 4-bit dense trunk; thinking on by default (no effort = max).
GLM_TF_DIR := $(HOME)/src/glm53-tensorfold-miaai

glm-tf: flush cache-flusher ## Launch GLM-5.3-Flash on TensorFold (MiaAI kit, 2-node, :8888, 1M ctx, served GLM-5.3-Flash-EXL3)
	cd $(GLM_TF_DIR) && ./start.sh

glm-tf-status: ## Is the TensorFold GLM lane serving (both ranks)?
	@docker ps --filter name=glm53-flash-tf --format "head:   {{.Names}} {{.Status}}"; ssh -o BatchMode=yes $(WORKER) docker ps --filter name=glm53-flash-tf --format "\"worker: {{.Names}} {{.Status}}\""

logs-glm-tf: ## Tail the TensorFold GLM rank-0 container
	docker logs -f --tail 200 glm53-flash-tf

stop-glm-tf: ## Stop the TensorFold GLM lane (both ranks)
	-cd $(GLM_TF_DIR) && ./stop.sh

# DeepSeek-V4.1-Flash NATIVE weights on three nodes: MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks
# @ cad252b (SGLang, AGPL-3.0; weights MIT), cloned 2026-09-29 as ~/src/ds41-sglang-miaai.
# Full local checkpoint on every node (NFS_SHARE=0: its exporter would fight mimo26-nfs /
# glm53-nfs for the head's nfsd), workers via the dsv41-local-weights bind volume; Engram
# rows packed per rank to ~/dsv41-engram (r?of3, beside the EXL3 lane's r?of2). 256K ctx,
# :8000, API_KEY in the kit's .env. HAZARD: the kit's stop.sh removes containers matching
# the SUBSTRING dsv41- — that includes the 2-node DS4.1 EXL3 lane's dsv41-exl3-*. Harmless
# while only one lane runs; never call it with the EXL3 lane up.
DS41X3_DIR := $(HOME)/src/ds41-sglang-miaai

ds41x3-build: ## One-time/after a pull: build the DS4.1 3-node SGLang overlay image on every node (pulls the pinned base)
	cd $(DS41X3_DIR) && ./start.sh build

ds41x3-pack: ## One-time/after a TP or weights change: pack each rank's Engram rows to NVMe (~10 min, ~63 GiB/node)
	cd $(DS41X3_DIR) && ./start.sh pack

ds41x3: flush3 cache-flusher3 ## Launch DeepSeek-V4.1-Flash native on THREE nodes (MiaAI SGLang kit, TP3, 256K ctx, served deepseek-v4.1-flash; boot ~13 min)
	cd $(DS41X3_DIR) && ./start.sh serve

ds41x3-status: ## Status of the DS4.1 3-node lane
	cd $(DS41X3_DIR) && ./start.sh status

logs-ds41x3: ## Tail the DS4.1 3-node head container
	cd $(DS41X3_DIR) && ./start.sh logs

stop-ds41x3: ## Stop the DS4.1 3-node lane (kit stop.sh; see the substring hazard above)
	-cd $(DS41X3_DIR) && ./stop.sh

## --- dry-run / VRAM fit estimate (no launch) ------------------------------

deepseek-dry: ## Estimate VRAM/context fit for DeepSeek-V4-Flash-Vision-Exp + DSpark
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES) --dry-run

qwen38fn-dry: ## Preflight the MiaAI Qwen3.8 kit: .env, worker SSH, weights on both nodes (no launch)
	cd $(QWEN38FN_DIR) && ./start.sh --no-download --no-launch && ./check-weights.sh

qwen-flash-dry: qwen-flash-sync ## Preflight the qwen-flash lane on both nodes: image, nvidia checkpoint, overlay files, RDMA, NICs (no launch)
	CHECK=1 $(QWEN_FLASH_ENV) bash $(QWEN_FLASH_LAUNCHER) 0
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'CHECK=1 $(QWEN_FLASH_ENV) bash ~/qwen-flash-tp2.sh 1'

mimo-dry: ## Preflight the MiMo lane (MiaAI kit doctor): image label + engine/memory preflight when the image exists, checkpoint files, worker ssh/docker/IB (launches nothing)
	cd $(MIMO26_DIR) && ./start.sh doctor

mimo-vllm-dry: mimo-vllm-sync ## Preflight the vLLM rollback lane on both nodes: model + staged patch files present, prints the docker run line (DRY_RUN, launches nothing)
	cd $(MIMO26_VLLM_DIR) && DRY_RUN=1 $(MIMO26_VLLM_ENV) bash launch/serve.sh 0
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'cd ~/src/mimo26-flash-tony && DRY_RUN=1 $(MIMO26_VLLM_ENV) $(MIMO26_VLLM_WORKER_ENV) bash launch/serve.sh 1 || [ $$? -eq 1 ]'

## --- lifecycle ------------------------------------------------------------

patch-sparkrun: ## Keep sparkrun's torch/gloo control plane off Wi-Fi (idempotent; re-run after `sparkrun update`)
	python3 tools/patch_sparkrun_wifi.py

# `flush` refuses to run when vm.swappiness is not 0 on BOTH nodes (added
# 2026-09-08). tonyd2wild's GLM README calls swappiness=0 mandatory on these
# 121 GiB unified-memory nodes and warns it does not survive a reboot: at the
# default (60) the kernel pages vLLM out mid-load (UVM livelock in their fleet).
# Here the reboot of 2026-09-07 05:32 silently reset both nodes to 60; every
# load then pushed 3-6 GB into swap, which opened earlyoom's `-s 80` swap gate,
# after which any dip under 2% MemAvailable kills vLLM (GLM's head rank was
# SIGTERM'd that way on 2026-09-08 16:00:35, 4 s before ready, while a
# checkpoint download ran on the head). Persisted now in
# /etc/sysctl.d/99-spark-swappiness.conf on both nodes; this guard catches the
# next time it is not. Fix: sudo sysctl vm.swappiness=0 on the node named.
flush: ## Drop the page cache on every node in WORKERS + the head (GB10 UMA: NVRM needs physically free memory for the KV slab); refuses if vm.swappiness != 0
	@bad=""; h=$$(cat /proc/sys/vm/swappiness); [ "$$h" = "0" ] || bad="head=$$h"; \
	for w in $(WORKERS); do s=$$(ssh -o BatchMode=yes -o ConnectTimeout=10 $$w cat /proc/sys/vm/swappiness 2>/dev/null || echo unreachable); \
	  [ "$$s" = "0" ] || bad="$$bad $$w=$$s"; done; \
	if [ -n "$$bad" ]; then \
	  echo "REFUSING TO LAUNCH: vm.swappiness must be 0 on every node, got:$$bad (see the comment above flush:)." >&2; \
	  echo "  fix: sudo sysctl vm.swappiness=0 && echo vm.swappiness=0 | sudo tee /etc/sysctl.d/99-spark-swappiness.conf   (on the node named)" >&2; exit 1; fi
	sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
	for w in $(WORKERS); do ssh -o BatchMode=yes -o ConnectTimeout=10 $$w 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null' || exit 1; done
	@echo "MemAvailable after flush:"; grep MemAvailable /proc/meminfo; for w in $(WORKERS); do ssh -o BatchMode=yes $$w grep MemAvailable /proc/meminfo; done

# tonyd2wild's cache_flusher.sh (GLM repo @ 050081d, NVIDIA KB 5776 remedy),
# adopted 2026-09-08 as tools/cache_flusher.sh: for 25 min after a launch it
# drops the page cache on each node whenever Cached passes 40 GiB, so NVRM can
# carve the KV slab out of physically free memory. `flush` only clears the
# cache once, before the 10-minute load refills it; upstream runs this
# alongside every boot. Every launch target depends on it. A second start
# replaces the first (pidfile); it exits on its own. Logs:
# ~/bench/cache-flusher-<host>.log on each node.
cache-flusher: ## Run tonyd2wild's page-cache flusher on the head + WORKERS for the next 25 min (flushes whenever Cached > 40 GiB)
	for w in $(WORKERS); do rsync -a tools/cache_flusher.sh $$w:cache_flusher.sh || exit 1; done
	nohup bash tools/cache_flusher.sh > /dev/null 2>&1 < /dev/null &
	for w in $(WORKERS); do ssh -o BatchMode=yes -o ConnectTimeout=10 $$w 'nohup bash ~/cache_flusher.sh > /dev/null 2>&1 < /dev/null &'; done
	@echo "cache flusher running on the head + $(WORKERS) for 25 min (logs: ~/bench/cache-flusher-<host>.log)"

stop-cache-flusher: ## Stop the page-cache flusher on every node (it also exits by itself after 25 min)
	-[ -f $(HOME)/.cache_flusher.pid ] && kill $$(cat $(HOME)/.cache_flusher.pid) 2>/dev/null; true
	-for w in $(WORKERS3); do ssh -o BatchMode=yes -o ConnectTimeout=10 $$w '[ -f ~/.cache_flusher.pid ] && kill $$(cat ~/.cache_flusher.pid) 2>/dev/null; true'; done

flush3: ## flush on all three nodes (the 3-node lanes' prerequisite)
	@$(MAKE) --no-print-directory flush WORKERS="$(WORKERS3)"

cache-flusher3: ## cache-flusher on all three nodes
	@$(MAKE) --no-print-directory cache-flusher WORKERS="$(WORKERS3)"

stop: ## Stop all workloads on the cluster (sparkrun lanes + the Qwen, qwen-flash, GLM-EXL3, DeepSeek and both MiMo kits, 2- and 3-node)
	$(SPARKRUN) stop --all --cluster $(CLUSTER)
	-cd $(DEEPSEEK_MIAAI_DIR) && ./stop-deepseek-v4-flash-dspark.sh
	-[ -f $(DS4_TP3_ENV) ] && cd $(DEEPSEEK_MIAAI_DIR) && ENV_FILE=$(DS4_TP3_ENV) ./stop-deepseek-v4-flash-dspark.sh
	-cd $(GLM_EXL3_DIR) && [ -f .env.tp3 ] && ./start-tp3.sh stop
	-[ -e $(JSPARK3_RUNTIME)/service.json ] && $(MAKE) --no-print-directory stop-jspark3
	-cd $(DS41X3_DIR) && [ -f .env ] && ./stop.sh
	-cd $(DS41_DIR) && ./start.sh stop
	-cd $(QWEN38FN_DIR) && ./stop.sh
	-cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh stop
	-docker rm -f vllm_qwen38fn 2>/dev/null
	-ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) docker rm -f vllm_qwen38fn 2>/dev/null
	-docker rm -f vllm_mimo 2>/dev/null
	-ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) docker rm -f vllm_mimo 2>/dev/null
	-cd $(MIMO26_DIR) && ./stop.sh --no-wait
	-$(MAKE) --no-print-directory stop-cache-flusher

stop-deepseek: ## Stop the DeepSeek lane (MiaAI kit; also clears the sparkrun rollback lane)
	-cd $(DEEPSEEK_MIAAI_DIR) && ./stop-deepseek-v4-flash-dspark.sh
	-$(SPARKRUN) stop $(DEEPSEEK_RECIPE) --cluster $(CLUSTER)

stop-ds41: ## Stop just the DeepSeek-V4.1-Flash EXL3 workload
	-cd $(DS41_DIR) && ./start.sh stop

stop-glm-exl3: ## Stop just the GLM-5.3-Flash EXL3 workload (glm53-exl3-head/-worker)
	cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh stop

stop-qwen38fn: ## Stop just the Qwen3.8-Flash-Next workload (MiaAI kit: vllm-fn on both nodes)
	cd $(QWEN38FN_DIR) && ./stop.sh

stop-qwen38fn-sglang: ## Stop the parked SGLang Qwen3.8 lane
	$(SPARKRUN) stop $(QWEN38FN_SGLANG_RECIPE) --cluster $(CLUSTER)

stop-qwen-flash: ## Stop just the qwen-flash lane (vllm_qwen38fn on both nodes)
	-docker rm -f vllm_qwen38fn
	-ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) docker rm -f vllm_qwen38fn

stop-mimo: ## Stop just the MiMo lane (MiaAI kit: mimo26-tp2-head/-worker, then waits for the driver to release the GPU on both nodes; exit 2 = a node never settled, do not serve)
	cd $(MIMO26_DIR) && ./start.sh stop

stop-mimo-vllm: ## Stop just the vLLM rollback lane (vllm_mimo on both nodes)
	-docker rm -f vllm_mimo
	-ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) docker rm -f vllm_mimo

status: ## Show running sparkrun containers (+ the vllm-fn / qwen-flash / glm53-exl3 / mimo26 / vllm_mimo kit containers, if any)
	$(SPARKRUN) status --cluster $(CLUSTER)
	@docker ps --filter name=vllm-fn --format 'vllm-fn (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=vllm-fn --format 'vllm-fn (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true
	@docker ps --filter name=glm53-exl3 --format 'glm53-exl3 (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=glm53-exl3 --format 'glm53-exl3 (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true
	@docker ps --filter name=vllm_qwen38fn --format 'qwen-flash (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=vllm_qwen38fn --format 'qwen-flash (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true
	@docker ps --filter name=mimo26-tp2-head --format 'mimo (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=mimo26-tp2-worker --format 'mimo (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true
	@docker ps --filter name=vllm_mimo --format 'mimo-vllm (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=vllm_mimo --format 'mimo-vllm (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER2) "docker ps --format '{{.Names}} (worker2): {{.Status}}  {{.Image}}'" 2>/dev/null || true

logs: ## Tail the running workload's logs (or a specific one: make logs TARGET=<job-id|recipe>)
	@target="$(TARGET)"; \
	if [ -z "$$target" ]; then \
 target=$$($(SPARKRUN) status --cluster $(CLUSTER) 2>/dev/null \
			| sed -n 's/.*\[\([0-9a-f]\{6,\}\)\].*/\1/p' | head -1); \
	fi; \
	if [ -z "$$target" ]; then \
 echo "No workload running on cluster '$(CLUSTER)' — nothing to tail (try 'make status')."; \
 exit 1; \
	fi; \
	echo "sparkrun logs $$target"; \
	$(SPARKRUN) logs $$target

logs-glm-exl3: ## Tail the GLM EXL3 head container (MiaAI kit launcher; `make logs` cannot see it)
	cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh logs

# PARKED 2026-09-08 with `make qwen38fn` (MiaAI kit); the stop targets stay live for cleanup.
#logs-qwen38fn: ## Tail the MiaAI Qwen3.8 head container (the kit is not a sparkrun job, so `make logs` cannot see it)
#	docker logs -f vllm-fn

logs-qwen-flash: ## Tail the qwen-flash head container (not a sparkrun job, so `make logs` cannot see it)
	docker logs -f vllm_qwen38fn

logs-mimo: ## Tail the MiMo head container (MiaAI kit; not a sparkrun job, so `make logs` cannot see it)
	cd $(MIMO26_DIR) && ./start.sh logs -f

logs-mimo-vllm: ## Tail the vLLM rollback lane's head container
	docker logs -f vllm_mimo

list: ## List available recipes
	$(SPARKRUN) list
