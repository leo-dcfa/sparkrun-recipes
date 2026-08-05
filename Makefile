# Makefile — convenience wrappers around `sparkrun run` for the recipes Leo runs
# day-to-day on the 2x DGX Spark (GB10) homelab.
#
#   make deepseek                         # launch DeepSeek-V4-Flash fp8+MTP, 1M ctx (local, MiaAI main lane)
#   make minimax                          # launch MiniMax-M2.7 NVFP4 (official)
#   make hy3                              # launch Hy3-295B NVFP4-W4A16 + MTP (local)
#   make deepseek-dspark                  # launch DeepSeek-V4-Flash + DSpark drafter, NVFP4 KV, 1M ctx (local)
#   make deepseek-0731                    # launch DeepSeek-V4-Flash-0731 + DSpark drafter, NVFP4 KV, 1M ctx (local)
#   make laguna                           # launch Laguna S 2.1 NVFP4 + DFlash drafter, 256K ctx (local, 1-node)
#   make mimo                             # launch MiMo-V2.5 NVFP4 Omni (multimodal), 1M ctx (local, 2-node)
#   make inkling                          # launch TML Inkling-Small NVFP4 (multimodal MoE), SGLang+DSpark, 1M ctx (local, 2-node)
#   make step                             # launch Step-3.7-Flash NVFP4, 256K ctx (local, 2-node)
#   make deepseek MAX_MODEL_LEN=500000    # override context length
#   make deepseek-dry                     # VRAM/fit estimate, no launch
#   make stop                             # stop everything on the cluster
#
# Recipes are registry-qualified (@registry/name); resolved via the enabled
# sparkrun registries — no local checkout needed.

SPARKRUN ?= sparkrun
CLUSTER  ?= leo-azl-2node

# Registry-qualified recipe identifiers.
# DeepSeek main lane — Spark Arena recipe (J C, 37.80 tok/s TG, 256K ctx,
# scitrera image, 2026-07-23). Replaced @experimental/deepseek4-flash-fp8-mtp-vllm.
# The MiaAI 1M-ctx variant (verified working, unbenchmarked) remains at
# recipes/deepseek-v4-flash-1m.yaml — swap the variable to use it.
DEEPSEEK_RECIPE       := recipes/deepseek-v4-flash.yaml
MINIMAX_RECIPE        := @official/minimax-m2.7-nvfp4-vllm
# (MiniMax-M3 REAP25 retired 2026-07-24 — was @experimental/minimax-m3-v0-nvfp4-2x-reap25.)
# (Spark Qwen lanes retired 2026-07-24 — were @official/qwen3.6-27b-fp8-mtp-vllm
#  and recipes/qwen3.6-35b-a3b-nvfp4-fast.yaml; the yaml stays in recipes/ for reference.)

# Local recipes (this repo) — run by file path, no registry needed.
HY3_RECIPE            := recipes/hy3-295b-nvfp4.yaml
# DeepSeek-V4-Flash + DSpark drafter — Spark Arena recipe (J C, 49.56 tok/s TG,
# 384K ctx, fp8 KV, production-3.7 image, 2026-07-23). Two earlier attempts at
# this lane failed the same day (tonyd Stage-C build; a hand-port of MiaAI's
# Anemll compose that garbled under sparkrun) — deploy/deepseek-dspark/ keeps
# the MiaAI compose deployment as a fallback (own start/stop scripts, do not
# launch via sparkrun).
DEEPSEEK_DSPARK_RECIPE := recipes/deepseek-v4-flash-dspark.yaml
DSPARK_DEPLOY_DIR      := deploy/deepseek-dspark
# DeepSeek-V4-Flash-0731 (official release) + DSpark drafter — tonyd2wild's
# forum lane (55.4 tok/s mean TG, 78.4 peak, 1M ctx, NVFP4 KV; adopted
# 2026-08-01). Runs the Patch-4 committed image
# vllm-dspark-runtime:dspark-nvfp4-stage-c-p4, built on BOTH nodes with
# recipes/mods/patch4_dspark_shared_experts.py — see the yaml header.
DEEPSEEK_0731_RECIPE   := recipes/deepseek-v4-flash-0731.yaml
# Poolside Laguna S 2.1 NVFP4 (118B-A8B MoE) + official DFlash drafter.
# Single-node / TP1 local recipe, 256K ctx, poolside_v1 parsers — see the yaml.
LAGUNA_RECIPE          := recipes/laguna-s-2.1-dflash.yaml
# FP8 variant — higher fidelity, but ~122 GB weights need BOTH Sparks (TP=2).
LAGUNA_FP8_RECIPE      := recipes/laguna-s-2.1-fp8-dflash.yaml
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
#   make deepseek MAX_MODEL_LEN=1000000 GPU_MEM=0.85
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

.PHONY: help deepseek minimax hy3 deepseek-dspark deepseek-dspark-compose deepseek-0731 laguna laguna-fp8 mimo inkling inkling-eugr step \
        deepseek-dry minimax-dry hy3-dry deepseek-dspark-dry deepseek-0731-dry laguna-dry laguna-fp8-dry mimo-dry inkling-dry step-dry \
        stop stop-deepseek stop-minimax stop-hy3 stop-deepseek-dspark stop-deepseek-0731 stop-laguna stop-laguna-fp8 stop-mimo stop-inkling stop-step \
        status logs list

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- launch ---------------------------------------------------------------

deepseek: ## Launch DeepSeek-V4-Flash fp8+MTP 1M ctx (local, 2-node, MiaAI main lane)
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES)

minimax: ## Launch MiniMax-M2.7 NVFP4 (official, 2-node)
	$(RUN) $(MINIMAX_RECIPE) $(OVERRIDES)

hy3: ## Launch Hy3-295B NVFP4-W4A16 + MTP (local, 2-node)
	$(RUN) $(HY3_RECIPE) $(OVERRIDES)

deepseek-dspark: ## Launch DeepSeek-V4-Flash + DSpark drafter (arena recipe, 2-node, fp8 KV, 384K ctx)
	$(RUN) $(DEEPSEEK_DSPARK_RECIPE) $(OVERRIDES)

deepseek-dspark-compose: ## Launch the MiaAI compose fallback for the DSpark lane (worker-first)
	cd $(DSPARK_DEPLOY_DIR) && PORT=8000 \
		API_URL=http://127.0.0.1:8000/v1/models \
		CHAT_URL=http://127.0.0.1:8000/v1/chat/completions \
		./start-deepseek-v4-flash-dspark.sh

deepseek-0731: ## Launch DeepSeek-V4-Flash-0731 + DSpark drafter (tonyd2wild lane, 2-node, NVFP4 KV, 1M ctx)
	$(RUN) $(DEEPSEEK_0731_RECIPE) $(OVERRIDES)

laguna: ## Launch Laguna S 2.1 NVFP4 + DFlash drafter (local, 1-node, 256K ctx)
	$(RUN) $(LAGUNA_RECIPE) $(OVERRIDES)

laguna-fp8: ## Launch Laguna S 2.1 FP8 + DFlash drafter (local, 2-node, 256K ctx)
	$(RUN) $(LAGUNA_FP8_RECIPE) $(OVERRIDES)

mimo: ## Launch MiMo-V2.5 NVFP4 Omni multimodal (local, 2-node, 450K ctx)
	$(RUN) $(MIMO_RECIPE) $(OVERRIDES)

inkling: ## Launch TML Inkling-Small NVFP4 multimodal MoE (local, 2-node, SGLang+DSpark, 1M ctx)
	$(RUN) $(INKLING_RECIPE) $(OVERRIDES)

inkling-eugr: ## Launch Inkling via eugr spark-vllm-docker fallback (2-node, 256K ctx)
	cd $(INKLING_DEPLOY_DIR) && ./run-recipe.sh inkling-small-nvfp4 -- --served-model-name inkling-small

step: ## Launch Step-3.7-Flash NVFP4 (local, 2-node, 256K ctx, MiaAI no-MTP lane)
	$(RUN) $(STEP_RECIPE) $(OVERRIDES)

## --- dry-run / VRAM fit estimate (no launch) ------------------------------

deepseek-dry: ## Estimate VRAM/context fit for DeepSeek
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES) --dry-run

minimax-dry: ## Estimate VRAM/context fit for MiniMax-M2.7 NVFP4
	$(RUN) $(MINIMAX_RECIPE) $(OVERRIDES) --dry-run

hy3-dry: ## Estimate VRAM/context fit for Hy3-295B NVFP4-W4A16
	$(RUN) $(HY3_RECIPE) $(OVERRIDES) --dry-run

deepseek-dspark-dry: ## Estimate VRAM/context fit for the DSpark arena recipe
	$(RUN) $(DEEPSEEK_DSPARK_RECIPE) $(OVERRIDES) --dry-run

deepseek-0731-dry: ## Estimate VRAM/context fit for DeepSeek-V4-Flash-0731 + DSpark
	$(RUN) $(DEEPSEEK_0731_RECIPE) $(OVERRIDES) --dry-run

laguna-dry: ## Estimate VRAM/context fit for Laguna S 2.1 NVFP4 + DFlash
	$(RUN) $(LAGUNA_RECIPE) $(OVERRIDES) --dry-run

laguna-fp8-dry: ## Estimate VRAM/context fit for Laguna S 2.1 FP8 + DFlash
	$(RUN) $(LAGUNA_FP8_RECIPE) $(OVERRIDES) --dry-run

mimo-dry: ## Estimate VRAM/context fit for MiMo-V2.5 NVFP4 Omni
	$(RUN) $(MIMO_RECIPE) $(OVERRIDES) --dry-run

inkling-dry: ## Estimate VRAM/context fit for TML Inkling-Small NVFP4 (SGLang+DSpark lane)
	$(RUN) $(INKLING_RECIPE) $(OVERRIDES) --dry-run

step-dry: ## Estimate VRAM/context fit for Step-3.7-Flash NVFP4
	$(RUN) $(STEP_RECIPE) $(OVERRIDES) --dry-run

## --- lifecycle ------------------------------------------------------------

stop: ## Stop all workloads on the cluster (sparkrun + dspark compose + inkling eugr lanes)
	$(SPARKRUN) stop --all --cluster $(CLUSTER)
	-cd $(DSPARK_DEPLOY_DIR) && ./stop-deepseek-v4-flash-dspark.sh 2>/dev/null
	-docker stop vllm_node 2>/dev/null
	-ssh 10.100.200.1 "docker stop vllm_node" 2>/dev/null

stop-deepseek: ## Stop just the DeepSeek workload
	$(SPARKRUN) stop $(DEEPSEEK_RECIPE) --cluster $(CLUSTER)

stop-minimax: ## Stop just the MiniMax-M2.7 NVFP4 workload
	$(SPARKRUN) stop $(MINIMAX_RECIPE) --cluster $(CLUSTER)

stop-hy3: ## Stop just the Hy3-295B NVFP4 workload
	$(SPARKRUN) stop $(HY3_RECIPE) --cluster $(CLUSTER)

stop-deepseek-dspark: ## Stop just the DeepSeek-V4-Flash + DSpark workload (sparkrun + compose fallback)
	-$(SPARKRUN) stop $(DEEPSEEK_DSPARK_RECIPE) --cluster $(CLUSTER)
	-cd $(DSPARK_DEPLOY_DIR) && ./stop-deepseek-v4-flash-dspark.sh 2>/dev/null

stop-deepseek-0731: ## Stop just the DeepSeek-V4-Flash-0731 + DSpark workload
	$(SPARKRUN) stop $(DEEPSEEK_0731_RECIPE) --cluster $(CLUSTER)

stop-laguna: ## Stop just the Laguna S 2.1 NVFP4 + DFlash workload
	$(SPARKRUN) stop $(LAGUNA_RECIPE) --cluster $(CLUSTER)

stop-laguna-fp8: ## Stop just the Laguna S 2.1 FP8 workload
	$(SPARKRUN) stop $(LAGUNA_FP8_RECIPE) --cluster $(CLUSTER)

stop-mimo: ## Stop just the MiMo-V2.5 NVFP4 Omni workload
	$(SPARKRUN) stop $(MIMO_RECIPE) --cluster $(CLUSTER)

stop-inkling: ## Stop just the Inkling-Small NVFP4 workload (sparkrun lane + eugr vllm_node fallback)
	-$(SPARKRUN) stop $(INKLING_RECIPE) --cluster $(CLUSTER)
	-docker stop vllm_node 2>/dev/null
	-ssh 10.100.200.1 "docker stop vllm_node" 2>/dev/null

stop-step: ## Stop just the Step-3.7-Flash NVFP4 workload
	$(SPARKRUN) stop $(STEP_RECIPE) --cluster $(CLUSTER)

status: ## Show running sparkrun containers
	$(SPARKRUN) status --cluster $(CLUSTER)

logs: ## Re-attach to running workload logs
	$(SPARKRUN) logs --cluster $(CLUSTER)

list: ## List available recipes
	$(SPARKRUN) list
