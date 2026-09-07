# Makefile — convenience wrappers around `sparkrun run` for the recipes Leo runs
# day-to-day on the 2x DGX Spark (GB10) homelab.
#
# make deepseek                         # launch DeepSeek-V4-Flash-Vision-Exp (native vision) + DSpark, NVFP4 KV, 1M ctx (local)
# make glm                              # launch GLM-5.3-Flash NVFP4 + DFlash2 k=7 spec decode (320B/18B-A multimodal MoE)
# make glm-exl3                         # launch GLM-5.3-Flash EXL3 4bpw + DFlash2 k=7, 1M ctx (Reederey87 kit, NOT sparkrun) — A/B lane vs `make glm`
# make qwen38fn                         # launch Qwen3.8-Flash-Next NVFP4, MiaAI vLLM TP2+EP+MTP3 kit, 262K ctx (2-node, NOT sparkrun)
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
# GLM-5.3-Flash EXL3/TR3 4bpw (brandonmusic quant, exllamav3 kernels built for
# sm_121a) + the same DFlash2 k=7 drafter — Reederey87's GB10-hardened fork of
# MiaAI's EXL3 kit, added 2026-09-07 as a SEPARATE lane to A/B against `make glm`,
# not as a replacement (github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark @
# 7d80e87; clone + this pair's .env at ~/src/glm53-exl3; every local value is
# marked LEO: in the .env). NOT a sparkrun recipe: like the Qwen kit it runs its
# own launcher (docker per node over SSH on the CX7 link). Why this fork and not
# MiaAI's original: fine-grained prefix-cache hits at 64-token grain (follow-up
# turns reuse 96-99% of the prompt, ~4 s -> ~1 s per turn), per-group KV
# retention (multi-session hits 0% -> 100%), a long-prefill fairness cap (short
# request behind a 240K read: 256 s -> ~7 s), memory-gated restarts and a JIT
# cache shape guard. MiaAI's README says its hits land only on 3584-token pages.
# What the A/B is for (tonyd2wild, same-clock, 2026-09-01): quality tie with the
# NVFP4 lane; NVFP4 faster on fresh prompts/prefill, EXL3 4x faster TTFT in
# multi-turn agent loops and 1M ctx (1.40M-token pool). Kit facts: image
# glm53-selfbuild is BUILT on the head from the kit Dockerfile (base
# vllm/vllm-openai:glm53-flash-arm64-cu130 @ sha256:905c0293 — already local;
# exllamav3 + fat-GEMM kernels compile in-image, ~40 min, needs the GPU idle for
# RAM), then shipped to the worker by start.sh. Weights brandonmusic/
# GLM-5.3-Flash-tr3-4bpw @ 1ae6d70 (~164 GiB on BOTH nodes; start.sh rsyncs the
# worker copy over CX7). Serving shape is the kit's PROD set: 1M ctx, MNBT 3584
# (= the hybrid page size; APC reads 0% otherwise), 4 seqs, KV pinned to
# 15414698763 bytes (NEVER raise), --no-async-scheduling, gmu 0.85 as boot gate,
# fp8_ds_mla KV, thinking on, vision on, served name glm-5.3-flash-exl3 on :8000.
# Local deviations: SERVED_MODEL_NAME/PORT, WORKER_SSH=leo@10.100.200.1, the
# cross-wired CX7 pins (head f0 / worker f1), HF_BIN="uvx hf" (no hf CLI here),
# HF_HUB_DISABLE_XET=1, GLM53_DEFAULT_REASONING_EFFORT empty (= template Max,
# matching `make glm`), and start.sh's --host 127.0.0.1 -> 0.0.0.0 (the kit
# binds loopback on purpose; LiteLLM lives on rtx-5090). prod-start.sh needs
# MemFree >= 90 GiB on both nodes, so `flush` runs first and the other lanes
# must be down (one model at a time). Gotchas from tonyd2wild's bring-up: the
# worker needs the FULL 164 GiB copy; ~/.cache/vllm-glm53-flash must be owned
# by leo on both nodes; first boot after any shape change is a long cold JIT
# (READY_TIMEOUT 4800). Status 2026-09-07: kit cloned, .env written, weights
# downloading — image build + first boot + A/B pending.
GLM_EXL3_DIR          := $(HOME)/src/glm53-exl3
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

.PHONY: help deepseek glm glm-exl3 glm-exl3-build qwen38fn qwen38fn-sglang \
        deepseek-dry glm-dry glm-exl3-dry qwen38fn-dry \
        stop stop-deepseek stop-glm stop-glm-exl3 stop-qwen38fn stop-qwen38fn-sglang \
        status logs logs-glm-exl3 logs-qwen38fn list flush patch-sparkrun

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- launch ---------------------------------------------------------------

deepseek: ## Launch DeepSeek-V4-Flash-Vision-Exp + DSpark k=5 (tonyd2wild vision port, 2-node, NVFP4 KV, 1M ctx)
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES)

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
glm: flush patch-sparkrun ## Launch GLM-5.3-Flash NVFP4 + DFlash2 k=7 spec decode (local, 2-node, 256K ctx)
	$(RUN) $(GLM_RECIPE) $(OVERRIDES)

glm-exl3: flush ## Launch GLM-5.3-Flash EXL3 4bpw + DFlash2 k=7 (Reederey87 kit, 2-node, 1M ctx) — A/B lane
	cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && local/prod-start.sh

glm-exl3-build: ## Build the EXL3 serving image glm53-selfbuild on the head (~40 min, GPU must be idle: RAM)
	cd $(GLM_EXL3_DIR) && docker build -t glm53-selfbuild . 2>&1 | tee ~/bench/glm53-exl3-build-$$(date +%Y%m%d-%H%M).log | tail -5

qwen38fn: flush ## Launch Qwen3.8-Flash-Next NVFP4 (MiaAI vLLM TP2+EP+MTP3 kit, 2-node, 262K ctx, bf16 KV)
	cd $(QWEN38FN_DIR) && ./start.sh --launch

qwen38fn-sglang: ## PARKED — tonyd2wild SGLang lane (dies on first real prefill); reference only
	$(RUN) $(QWEN38FN_SGLANG_RECIPE) $(OVERRIDES)

## --- dry-run / VRAM fit estimate (no launch) ------------------------------

deepseek-dry: ## Estimate VRAM/context fit for DeepSeek-V4-Flash-Vision-Exp + DSpark
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES) --dry-run

glm-dry: ## Estimate VRAM/context fit for GLM-5.3-Flash NVFP4 + DFlash2
	$(RUN) $(GLM_RECIPE) $(OVERRIDES) --dry-run

glm-exl3-dry: ## Validate the EXL3 kit config (.env, fabric pins, GID tables) without launching
	cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh validate

qwen38fn-dry: ## Preflight the MiaAI Qwen3.8 kit: .env, worker SSH, weights on both nodes (no launch)
	cd $(QWEN38FN_DIR) && ./start.sh --no-download --no-launch && ./check-weights.sh

## --- lifecycle ------------------------------------------------------------

patch-sparkrun: ## Keep sparkrun's torch/gloo control plane off Wi-Fi (idempotent; re-run after `sparkrun update`)
	python3 tools/patch_sparkrun_wifi.py

flush: ## Drop the page cache on both nodes (GB10 UMA: NVRM needs physically free memory for the KV slab)
	sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'
	@echo "MemAvailable after flush:"; grep MemAvailable /proc/meminfo; ssh -o BatchMode=yes $(WORKER) grep MemAvailable /proc/meminfo

stop: ## Stop all workloads on the cluster (sparkrun lanes + the Qwen and GLM-EXL3 kits)
	$(SPARKRUN) stop --all --cluster $(CLUSTER)
	-cd $(QWEN38FN_DIR) && ./stop.sh
	-cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh stop

stop-deepseek: ## Stop just the DeepSeek-V4-Flash-Vision-Exp + DSpark workload
	$(SPARKRUN) stop $(DEEPSEEK_RECIPE) --cluster $(CLUSTER)

stop-glm: ## Stop just the GLM-5.3-Flash NVFP4 + DFlash2 workload
	$(SPARKRUN) stop $(GLM_RECIPE) --cluster $(CLUSTER)

stop-glm-exl3: ## Stop just the GLM-5.3-Flash EXL3 workload (glm53-exl3-head/-worker)
	cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh stop

stop-qwen38fn: ## Stop just the Qwen3.8-Flash-Next workload (MiaAI kit: vllm-fn on both nodes)
	cd $(QWEN38FN_DIR) && ./stop.sh

stop-qwen38fn-sglang: ## Stop the parked SGLang Qwen3.8 lane
	$(SPARKRUN) stop $(QWEN38FN_SGLANG_RECIPE) --cluster $(CLUSTER)

status: ## Show running sparkrun containers (+ the MiaAI vllm-fn containers, if any)
	$(SPARKRUN) status --cluster $(CLUSTER)
	@docker ps --filter name=vllm-fn --format 'vllm-fn (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=vllm-fn --format 'vllm-fn (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true
	@docker ps --filter name=glm53-exl3 --format 'glm53-exl3 (head):   {{.Status}}  {{.Image}}' 2>/dev/null || true
	@ssh -o BatchMode=yes -o ConnectTimeout=5 $(WORKER) "docker ps --filter name=glm53-exl3 --format 'glm53-exl3 (worker): {{.Status}}  {{.Image}}'" 2>/dev/null || true

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

logs-glm-exl3: ## Tail the GLM EXL3 head container (kit launcher; `make logs` cannot see it)
	cd $(GLM_EXL3_DIR) && set -a && . ./.env && set +a && ./start.sh logs

logs-qwen38fn: ## Tail the MiaAI Qwen3.8 head container (the kit is not a sparkrun job, so `make logs` cannot see it)
	docker logs -f vllm-fn

list: ## List available recipes
	$(SPARKRUN) list
