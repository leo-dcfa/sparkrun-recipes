# Makefile — convenience wrappers around `sparkrun run` for the recipes Leo runs
# day-to-day on the 2x DGX Spark (GB10) homelab.
#
# make deepseek                         # launch DeepSeek-V4-Flash-Vision-Exp (native vision) + DSpark, NVFP4 KV, 1M ctx (local)
# make glm                              # launch GLM-5.3-Flash NVFP4 + DFlash2 k=7 spec decode (320B/18B-A multimodal MoE)
# make qwen38fn                         # launch Qwen3.8-Flash-Next NVFP4, SGLang TP2 + NEXTN spec decode, 262K ctx (local, 2-node)
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
# Qwen3.8-Flash-Next NVFP4 (125B-A3B hybrid MoE + 51B PLE + MTP head,
# multimodal) — tonyd2wild's SGLang TP2 lane (NEXTN spec decode, decode CUDA
# graphs, 600K-token KV pin, thinking OFF server-side), adopted 2026-09-03 on
# their STAGED Triton-varlen image built locally on BOTH nodes
# (docker/Dockerfile.qwen38fn-sm121-triton-varlen -> qwen38fn-sglang:
# sm121-triton-varlen-local). UNVERIFIED on this host beyond the image build and
# a dry run — see the yaml header for upstream's promotion checklist and the
# history of the two earlier (non-Makefile) Qwen3.8-Flash-Next lanes; the
# hand-run vLLM launcher at ~/src/qwen38-flashnext-vllm stays the fallback.
# Upstream drops page cache on both nodes before launch (GB10 UMA):
#   sync; echo 3 | sudo tee /proc/sys/vm/drop_caches   (head AND worker)
QWEN38FN_RECIPE        := recipes/qwen3.8-flash-next-sglang.yaml

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

.PHONY: help deepseek glm qwen38fn \
        deepseek-dry glm-dry qwen38fn-dry \
        stop stop-deepseek stop-glm stop-qwen38fn \
        status logs list flush patch-sparkrun

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

qwen38fn: ## Launch Qwen3.8-Flash-Next NVFP4 (local, 2-node, SGLang TP2 + NEXTN, 262K ctx, tonyd2wild lane)
	$(RUN) $(QWEN38FN_RECIPE) $(OVERRIDES)

## --- dry-run / VRAM fit estimate (no launch) ------------------------------

deepseek-dry: ## Estimate VRAM/context fit for DeepSeek-V4-Flash-Vision-Exp + DSpark
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES) --dry-run

glm-dry: ## Estimate VRAM/context fit for GLM-5.3-Flash NVFP4 + DFlash2
	$(RUN) $(GLM_RECIPE) $(OVERRIDES) --dry-run

qwen38fn-dry: ## Estimate VRAM/context fit for Qwen3.8-Flash-Next NVFP4
	$(RUN) $(QWEN38FN_RECIPE) $(OVERRIDES) --dry-run

## --- lifecycle ------------------------------------------------------------

patch-sparkrun: ## Keep sparkrun's torch/gloo control plane off Wi-Fi (idempotent; re-run after `sparkrun update`)
	python3 tools/patch_sparkrun_wifi.py

flush: ## Drop the page cache on both nodes (GB10 UMA: NVRM needs physically free memory for the KV slab)
	sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
	ssh -o BatchMode=yes -o ConnectTimeout=10 $(WORKER) 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'
	@echo "MemAvailable after flush:"; grep MemAvailable /proc/meminfo; ssh -o BatchMode=yes $(WORKER) grep MemAvailable /proc/meminfo

stop: ## Stop all workloads on the cluster
	$(SPARKRUN) stop --all --cluster $(CLUSTER)

stop-deepseek: ## Stop just the DeepSeek-V4-Flash-Vision-Exp + DSpark workload
	$(SPARKRUN) stop $(DEEPSEEK_RECIPE) --cluster $(CLUSTER)

stop-glm: ## Stop just the GLM-5.3-Flash NVFP4 + DFlash2 workload
	$(SPARKRUN) stop $(GLM_RECIPE) --cluster $(CLUSTER)

stop-qwen38fn: ## Stop just the Qwen3.8-Flash-Next NVFP4 workload
	$(SPARKRUN) stop $(QWEN38FN_RECIPE) --cluster $(CLUSTER)

status: ## Show running sparkrun containers
	$(SPARKRUN) status --cluster $(CLUSTER)

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

list: ## List available recipes
	$(SPARKRUN) list
