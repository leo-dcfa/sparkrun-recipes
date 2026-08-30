# Makefile — convenience wrappers around `sparkrun run` for the recipes Leo runs
# day-to-day on the 2x DGX Spark (GB10) homelab.
#
# make hy3                              # launch Hy3-295B NVFP4-W4A16 + MTP (local)
# make deepseek                         # launch DeepSeek-V4-Flash-0731 + DSpark drafter, NVFP4 KV, 1M ctx (local)
# make mimo                             # launch MiMo-V2.5 NVFP4 Omni (multimodal), 1M ctx (local, 2-node)
# make inkling                          # launch TML Inkling-Small NVFP4 (multimodal MoE), SGLang+DSpark, 1M ctx (local, 2-node)
# make step                             # launch Step-3.7-Flash NVFP4, 256K ctx (local, 2-node)
# make glm                              # launch GLM-5.3-Flash NVFP4 + DFlash2 k=7 spec decode (320B/18B-A multimodal MoE)
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
# and recipes/qwen3.6-35b-a3b-nvfp4-fast.yaml; the yaml stays in recipes/ for reference.)

# Local recipes (this repo) — run by file path, no registry needed.
HY3_RECIPE            := recipes/hy3-295b-nvfp4.yaml
# DeepSeek-V4-Flash-0731 (official release) + DSpark drafter — tonyd2wild's
# forum lane (adopted 2026-08-01; E15 STACK campaign winner adopted 2026-08-29:
# k=5 + Patch A + fused Markov argmax @ gmu 0.78). Measured here 2026-08-29:
# 64.4 tok/s battery mean / 91 peak (was 53.2/64.9 on k=3), c4 aggregate 172
# (was 124), 1M ctx, NVFP4 KV. Runs the Patch-4+A committed image
# vllm-dspark-runtime:dspark-nvfp4-stage-c-p4a, built on BOTH nodes with
# docker/Dockerfile.dspark-0731-p4a (worker builds from ~/staging-p4a) — see
# the yaml header.
DEEPSEEK_RECIPE       := recipes/deepseek-v4-flash-0731.yaml
# GLM-5.3-Flash NVFP4 + DFlash2 speculative decoding (320B total / 18B active,
# natively multimodal MoE). tonyd2wild's lane, adopted 2026-08-30 — see the yaml
# header for the full why. Measured here that day, single-stream, warm, greedy,
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
# MiMo-V2.5 NVFP4 Omni (multimodal) — MiaAI's dual-Spark Ray deploy, 1M ctx.
# Replaces the never-wired mimo-v2.5-dflash.yaml as the MiMo entry point.
MIMO_RECIPE            := recipes/mimo-v2.5-omni.yaml
# TML Inkling-Small NVFP4 (276B-A12B multimodal MoE) — MiaAI/drowzeys SGLang +
# DSpark champion lane, 1M ctx (adopted 2026-08-06, unverified on this host;
# see the yaml header for the pull-both-models/pull-image prep). The old parked
# vLLM lane is kept at recipes/inkling-small-nvfp4.yaml.vllm-parked; eugr's
# spark-vllm-docker deploy (256K ctx) stays available as `make inkling-eugr`
# until this lane is verified.
INKLING_RECIPE         := recipes/inkling-small-nvfp4.yaml
INKLING_DEPLOY_DIR     := $(HOME)/spark-vllm-docker
# StepFun Step-3.7-Flash NVFP4 — MiaAI's dual-Spark no-MTP lane, 256K ctx
# (adopted 2026-08-06, unverified on this host). Needs the local image built on
# BOTH nodes first: see docker/Dockerfile.stepfun37-procps.
STEP_RECIPE            := recipes/step-3.7-flash-nvfp4.yaml

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

.PHONY: help hy3 deepseek glm mimo inkling inkling-eugr step \
        hy3-dry deepseek-dry glm-dry mimo-dry inkling-dry step-dry \
        stop stop-hy3 stop-deepseek stop-glm stop-mimo stop-inkling stop-step \
        status logs list

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- launch ---------------------------------------------------------------

hy3: ## Launch Hy3-295B NVFP4-W4A16 + MTP (local, 2-node)
	$(RUN) $(HY3_RECIPE) $(OVERRIDES)

deepseek: ## Launch DeepSeek-V4-Flash-0731 + DSpark drafter (tonyd2wild lane, 2-node, NVFP4 KV, 1M ctx)
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES)

glm: ## Launch GLM-5.3-Flash NVFP4 + DFlash2 k=7 spec decode (local, 2-node, 256K ctx)
	$(RUN) $(GLM_RECIPE) $(OVERRIDES)

mimo: ## Launch MiMo-V2.5 NVFP4 Omni multimodal (local, 2-node, 450K ctx)
	$(RUN) $(MIMO_RECIPE) $(OVERRIDES)

inkling: ## Launch TML Inkling-Small NVFP4 multimodal MoE (local, 2-node, SGLang+DSpark, 1M ctx)
	$(RUN) $(INKLING_RECIPE) $(OVERRIDES)

inkling-eugr: ## Launch Inkling via eugr spark-vllm-docker fallback (2-node, 256K ctx)
	cd $(INKLING_DEPLOY_DIR) && ./run-recipe.sh inkling-small-nvfp4 -- --served-model-name inkling-small

step: ## Launch Step-3.7-Flash NVFP4 (local, 2-node, 256K ctx, MiaAI no-MTP lane)
	$(RUN) $(STEP_RECIPE) $(OVERRIDES)

## --- dry-run / VRAM fit estimate (no launch) ------------------------------

hy3-dry: ## Estimate VRAM/context fit for Hy3-295B NVFP4-W4A16
	$(RUN) $(HY3_RECIPE) $(OVERRIDES) --dry-run

deepseek-dry: ## Estimate VRAM/context fit for DeepSeek-V4-Flash-0731 + DSpark
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES) --dry-run

glm-dry: ## Estimate VRAM/context fit for GLM-5.3-Flash NVFP4 + DFlash2
	$(RUN) $(GLM_RECIPE) $(OVERRIDES) --dry-run

mimo-dry: ## Estimate VRAM/context fit for MiMo-V2.5 NVFP4 Omni
	$(RUN) $(MIMO_RECIPE) $(OVERRIDES) --dry-run

inkling-dry: ## Estimate VRAM/context fit for TML Inkling-Small NVFP4 (SGLang+DSpark lane)
	$(RUN) $(INKLING_RECIPE) $(OVERRIDES) --dry-run

step-dry: ## Estimate VRAM/context fit for Step-3.7-Flash NVFP4
	$(RUN) $(STEP_RECIPE) $(OVERRIDES) --dry-run

## --- lifecycle ------------------------------------------------------------

stop: ## Stop all workloads on the cluster (sparkrun + inkling eugr lanes)
	$(SPARKRUN) stop --all --cluster $(CLUSTER)
	@docker stop vllm_node >/dev/null 2>&1 && echo "stopped eugr vllm_node (head)" || true
	@ssh 10.100.200.1 "docker stop vllm_node" >/dev/null 2>&1 && echo "stopped eugr vllm_node (worker)" || true

stop-hy3: ## Stop just the Hy3-295B NVFP4 workload
	$(SPARKRUN) stop $(HY3_RECIPE) --cluster $(CLUSTER)

stop-deepseek: ## Stop just the DeepSeek-V4-Flash-0731 + DSpark workload
	$(SPARKRUN) stop $(DEEPSEEK_RECIPE) --cluster $(CLUSTER)

stop-glm: ## Stop just the GLM-5.3-Flash NVFP4 + DFlash2 workload
	$(SPARKRUN) stop $(GLM_RECIPE) --cluster $(CLUSTER)

stop-mimo: ## Stop just the MiMo-V2.5 NVFP4 Omni workload
	$(SPARKRUN) stop $(MIMO_RECIPE) --cluster $(CLUSTER)

stop-inkling: ## Stop just the Inkling-Small NVFP4 workload (sparkrun lane + eugr vllm_node fallback)
	-$(SPARKRUN) stop $(INKLING_RECIPE) --cluster $(CLUSTER)
	@docker stop vllm_node >/dev/null 2>&1 && echo "stopped eugr vllm_node (head)" || true
	@ssh 10.100.200.1 "docker stop vllm_node" >/dev/null 2>&1 && echo "stopped eugr vllm_node (worker)" || true

stop-step: ## Stop just the Step-3.7-Flash NVFP4 workload
	$(SPARKRUN) stop $(STEP_RECIPE) --cluster $(CLUSTER)

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
