# Makefile — convenience wrappers around `sparkrun run` for the recipes Leo runs
# day-to-day on the 2x DGX Spark (GB10) homelab.
#
# make deepseek                         # launch DeepSeek-V4-Flash-Vision-Exp + DSpark MTP=6, NVFP4 KV, 1M ctx (MiaAI-Lab kit, NOT sparkrun)
# make deepseek-sparkrun                # ROLLBACK lane: the tonyd2wild sparkrun recipe this replaced (k=5, 12 seqs)
# make ds41                             # launch DeepSeek-V4.1-Flash EXL3 2.9bpw (MiaAI-Lab kit, 552B, text-only, 600K ctx) -- NEEDS WEIGHTS, see below
# make glm                              # launch GLM-5.3-Flash NVFP4 + DFlash2 k=7 spec decode (320B/18B-A multimodal MoE)
# make glm-exl3                         # launch GLM-5.3-Flash EXL3 4bpw + DFlash2 k=7, 850K ctx (MiaAI-Lab kit, NOT sparkrun) — A/B lane vs `make glm`
# make qwen38fn                         # PARKED 2026-09-08 (commented out below; MiaAI vLLM TP2+EP+MTP3 kit) — superseded by `make qwen-flash`
# make qwen-flash                       # launch Qwen3.8-Flash-Next NVFP4 (nvidia ckpt), tonyd2wild vLLM TP2 SPEED lane, 262K ctx, thinking ON (2-node, NOT sparkrun)
# make qwen-flash-no-thinking           # same lane, enable_thinking false server-side
# make deepseek MAX_MODEL_LEN=500000    # override context length
# make deepseek-dry                     # VRAM/fit estimate, no launch
# make stop                             # stop everything on the cluster
#
# Recipes are registry-qualified (@registry/name); resolved via the enabled
# sparkrun registries — no local checkout needed.

SPARKRUN ?= sparkrun
CLUSTER  ?= leo-azl-2node

# sparkrun lives in ~/.local/bin, which non-interactive shells (e.g.
# `ssh spark-f31f make -C sparkrun-recipes glm`) don't have on PATH — without
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
# DGX-Sparks @ 8530568, clone + this pair's .env at ~/src/ds41-exl3-miaai
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
# GLM-5.3-Flash NVFP4 + DFlash2 speculative decoding (320B total / 18B active,
# natively multimodal MoE). tonyd2wild's lane, adopted 2026-08-30, re-synced
# 2026-09-03 (KV pin 3 -> 6 GiB: 0 preemptions under load vs 6, pool 310K ->
# 679K tokens; --max-num-batched-tokens 8192) — see the yaml header for the
# full why. Measured here that day, single-stream, warm, greedy,
# 500-token code prompt: 34.5 tok/s thinking-on (46.8% draft acceptance), 41.3
# thinking-off (60.0%), against the previous MTP-4 lane's ~21. Speculative
# decoding verifies every drafted token against the full model, so accepted
# output is bit-identical — speed, not a quality trade. Also switches the
# checkpoint to RedHatAI (the ModelOpt build emits intermittent corrupted token
# IDs on GB10, vLLM #54150, which desync tool-call parsing) and fixes vision,
# which never actually worked on the old lane for want of a multimodal template.
#
# (The MTP-4 lane was retired the same day: it was `make glm` on the MiaAI v8
#  image glm53-flash-sm121:v8 with LibertAIDAI weights. recipes/glm-5.3-flash-nvfp4.yaml
#  and that image are both kept for rollback — point GLM_RECIPE back at the yaml.)
GLM_RECIPE            := recipes/glm-5.3-flash-dflash2.yaml
# GLM-5.3-Flash EXL3/TR3 4bpw + DFlash2 k=7 — the A/B lane against `make glm`.
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
# ROLLBACK: the fork clone is untouched at ~/src/glm53-exl3 (its .env, and the
# unmodified start.sh which still binds 127.0.0.1). Point GLM_EXL3_DIR back at
# it, restore `local/prod-start.sh` as the launch command below, and re-add the
# glm-exl3-build / glm-exl3-dry targets from git history.
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

RUN := $(SPARKRUN) run --cluster $(CLUSTER)

# The worker node (node_1), addressed over the cluster link the way sparkrun does.
WORKER ?= 10.100.200.1

.PHONY: help deepseek deepseek-sparkrun ds41 ds41-status logs-ds41 stop-ds41 glm glm-exl3 qwen38fn-sglang qwen-flash qwen-flash-no-thinking qwen-flash-sync \
        deepseek-dry glm-dry qwen-flash-dry \
        stop stop-deepseek stop-glm stop-glm-exl3 stop-qwen38fn stop-qwen38fn-sglang stop-qwen-flash \
        status logs logs-glm-exl3 logs-qwen-flash list flush patch-sparkrun cache-flusher stop-cache-flusher
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
#   docker inspect <node_0> | grep -E "VLLM_HOST_IP|GLOO_SOCKET_IFNAME" -> 10.100.200.2 / enp1s0f0np0
#   grep mq_connect_ip /tmp/sparkrun_serve.log (in-container)         -> 10.100.200.2, not 192.168.0.120
# Cabling the 10GbE ports would make the patch redundant (wired default route).
glm: flush patch-sparkrun cache-flusher ## Launch GLM-5.3-Flash NVFP4 + DFlash2 k=7 spec decode (local, 2-node, 256K ctx)
	$(RUN) $(GLM_RECIPE) $(OVERRIDES)

glm-exl3: flush cache-flusher ## Launch GLM-5.3-Flash EXL3 4bpw + DFlash2 k=7 (MiaAI-Lab kit, 2-node, 850K ctx, FP8 dense + adaptive-k) — A/B lane
	cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh start

# PARKED 2026-09-08 — superseded by `make qwen-flash`; uncomment to roll back to the MiaAI kit.
#qwen38fn: flush ## Launch Qwen3.8-Flash-Next NVFP4 (MiaAI vLLM TP2+EP+MTP3 kit, 2-node, 262K ctx, bf16 KV)
#	cd $(QWEN38FN_DIR) && ./start.sh --launch

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

## --- dry-run / VRAM fit estimate (no launch) ------------------------------

deepseek-dry: ## Estimate VRAM/context fit for DeepSeek-V4-Flash-Vision-Exp + DSpark
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES) --dry-run

glm-dry: ## Estimate VRAM/context fit for GLM-5.3-Flash NVFP4 + DFlash2
	$(RUN) $(GLM_RECIPE) $(OVERRIDES) --dry-run

# PARKED 2026-09-08 with `make qwen38fn` (MiaAI kit); uncomment together.
#qwen38fn-dry: ## Preflight the MiaAI Qwen3.8 kit: .env, worker SSH, weights on both nodes (no launch)
#	cd $(QWEN38FN_DIR) && ./start.sh --no-download --no-launch && ./check-weights.sh

qwen-flash-dry: qwen-flash-sync ## Preflight the qwen-flash lane on both nodes: image, nvidia checkpoint, overlay files, RDMA, NICs (no launch)
	CHECK=1 $(QWEN_FLASH_ENV) bash $(QWEN_FLASH_LAUNCHER) 0
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'CHECK=1 $(QWEN_FLASH_ENV) bash ~/qwen-flash-tp2.sh 1'

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
flush: ## Drop the page cache on both nodes (GB10 UMA: NVRM needs physically free memory for the KV slab); refuses if vm.swappiness != 0
	@h=$$(cat /proc/sys/vm/swappiness); w=$$(ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) cat /proc/sys/vm/swappiness 2>/dev/null || echo unreachable); \
	if [ "$$h" != "0" ] || [ "$$w" != "0" ]; then \
	  echo "REFUSING TO LAUNCH: vm.swappiness is $$h on the head and $$w on the worker; both must be 0 (see the comment above flush:)." >&2; \
	  echo "  fix: sudo sysctl vm.swappiness=0 && echo vm.swappiness=0 | sudo tee /etc/sysctl.d/99-spark-swappiness.conf   (on the node named)" >&2; exit 1; fi
	sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'
	@echo "MemAvailable after flush:"; grep MemAvailable /proc/meminfo; ssh -o BatchMode=yes $(WORKER) grep MemAvailable /proc/meminfo

# tonyd2wild's cache_flusher.sh (GLM repo @ 050081d, NVIDIA KB 5776 remedy),
# adopted 2026-09-08 as tools/cache_flusher.sh: for 25 min after a launch it
# drops the page cache on each node whenever Cached passes 40 GiB, so NVRM can
# carve the KV slab out of physically free memory. `flush` only clears the
# cache once, before the 10-minute load refills it; upstream runs this
# alongside every boot. Every launch target depends on it. A second start
# replaces the first (pidfile); it exits on its own. Logs:
# ~/bench/cache-flusher-<host>.log on each node.
cache-flusher: ## Run tonyd2wild's page-cache flusher on both nodes for the next 25 min (flushes whenever Cached > 40 GiB)
	rsync -a tools/cache_flusher.sh $(WORKER):cache_flusher.sh
	nohup bash tools/cache_flusher.sh > /dev/null 2>&1 < /dev/null &
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'nohup bash ~/cache_flusher.sh > /dev/null 2>&1 < /dev/null &'
	@echo "cache flusher running on both nodes for 25 min (logs: ~/bench/cache-flusher-<host>.log)"

stop-cache-flusher: ## Stop the page-cache flusher on both nodes (it also exits by itself after 25 min)
	-[ -f $(HOME)/.cache_flusher.pid ] && kill $$(cat $(HOME)/.cache_flusher.pid) 2>/dev/null; true
	-ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) '[ -f ~/.cache_flusher.pid ] && kill $$(cat ~/.cache_flusher.pid) 2>/dev/null; true'

stop: ## Stop all workloads on the cluster (sparkrun lanes + the Qwen, qwen-flash, GLM-EXL3 and DeepSeek kits)
	$(SPARKRUN) stop --all --cluster $(CLUSTER)
	-cd $(DEEPSEEK_MIAAI_DIR) && ./stop-deepseek-v4-flash-dspark.sh
	-cd $(DS41_DIR) && ./start.sh stop
	-cd $(QWEN38FN_DIR) && ./stop.sh
	-cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh stop
	-docker rm -f vllm_qwen38fn 2>/dev/null
	-ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) docker rm -f vllm_qwen38fn 2>/dev/null
	-$(MAKE) --no-print-directory stop-cache-flusher

stop-deepseek: ## Stop the DeepSeek lane (MiaAI kit; also clears the sparkrun rollback lane)
	-cd $(DEEPSEEK_MIAAI_DIR) && ./stop-deepseek-v4-flash-dspark.sh
	-$(SPARKRUN) stop $(DEEPSEEK_RECIPE) --cluster $(CLUSTER)

stop-glm: ## Stop just the GLM-5.3-Flash NVFP4 + DFlash2 workload
	$(SPARKRUN) stop $(GLM_RECIPE) --cluster $(CLUSTER)

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

status: ## Show running sparkrun containers (+ the vllm-fn / qwen-flash / glm53-exl3 kit containers, if any)
	$(SPARKRUN) status --cluster $(CLUSTER)
	@docker ps --filter name=vllm-fn --format 'vllm-fn (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=vllm-fn --format 'vllm-fn (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true
	@docker ps --filter name=glm53-exl3 --format 'glm53-exl3 (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=glm53-exl3 --format 'glm53-exl3 (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true
	@docker ps --filter name=vllm_qwen38fn --format 'qwen-flash (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=vllm_qwen38fn --format 'qwen-flash (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true

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

list: ## List available recipes
	$(SPARKRUN) list
