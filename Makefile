# Makefile — convenience wrappers around `sparkrun run` for the recipes Leo runs
# day-to-day on the 2x DGX Spark (GB10) homelab.
#
#   make deepseek                         # launch DeepSeek-V4-Flash (fp8 + MTP)
#   make minimax                          # launch MiniMax-M2.7 NVFP4 (official)
#   make m3                               # launch MiniMax-M3 v0 NVFP4 REAP-25 (experimental, sglang)
#   make qwen                             # launch Qwen3.6-27B FP8+MTP (official)
#   make qwen35                           # launch Qwen3.6-35B-A3B NVFP4-Fast + MTP, 1M ctx (local)
#   make hy3                              # launch Hy3-295B NVFP4-W4A16 + MTP (local)
#   make deepseek-dspark                  # launch DeepSeek-V4-Flash + DSpark drafter (local)
#   make deepseek MAX_MODEL_LEN=1000000   # override context length
#   make deepseek-dry                     # VRAM/fit estimate, no launch
#   make stop                             # stop everything on the cluster
#
# Recipes are registry-qualified (@registry/name); resolved via the enabled
# sparkrun registries — no local checkout needed.

SPARKRUN ?= sparkrun
CLUSTER  ?= leo-azl-2node

# Registry-qualified recipe identifiers.
DEEPSEEK_RECIPE       := @experimental/deepseek4-flash-fp8-mtp-vllm
MINIMAX_RECIPE        := @official/minimax-m2.7-nvfp4-vllm
# MiniMax-M3 v0 NVFP4, 25% REAP expert-pruned (~93.5B params). sglang runtime,
# cross-node TP2. Experimental / unverified — see README.
M3_RECIPE             := @experimental/minimax-m3-v0-nvfp4-2x-reap25
# Qwen3.6-27B FP8 with multi-token prediction (MTP) for faster decode.
# Single-node / TP1 recipe.
QWEN_RECIPE           := @official/qwen3.6-27b-fp8-mtp-vllm
# Qwen3.6-35B-A3B Unsloth NVFP4-Fast: flashinfer_b12x kernels + MTP, 1M ctx via
# YaRN. Single-node / TP1 local recipe — dry-run validated only, see README.
QWEN35_RECIPE         := recipes/qwen3.6-35b-a3b-nvfp4-fast.yaml

# Local recipes (this repo) — run by file path, no registry needed.
HY3_RECIPE            := recipes/hy3-295b-nvfp4.yaml
# DeepSeek-V4-Flash + DSpark drafter (eugr's vllm-node image). Distinct from the
# published DEEPSEEK_RECIPE above (which is fp8 + MTP). fp8 KV, cross-node TP2.
DEEPSEEK_DSPARK_RECIPE := recipes/deepseek-v4-flash-dspark.yaml

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

.PHONY: help deepseek minimax m3 qwen qwen35 hy3 deepseek-dspark \
        deepseek-dry minimax-dry m3-dry qwen-dry qwen35-dry hy3-dry deepseek-dspark-dry \
        stop stop-deepseek stop-minimax stop-m3 stop-qwen stop-qwen35 stop-hy3 stop-deepseek-dspark \
        status logs list

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- launch ---------------------------------------------------------------

deepseek: ## Launch DeepSeek-V4-Flash (experimental, fp8 + MTP)
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES)

minimax: ## Launch MiniMax-M2.7 NVFP4 (official, 2-node)
	$(RUN) $(MINIMAX_RECIPE) $(OVERRIDES)

m3: ## Launch MiniMax-M3 v0 NVFP4 REAP-25 (experimental, sglang, 2-node)
	$(RUN) $(M3_RECIPE) $(OVERRIDES)

qwen: ## Launch Qwen3.6-27B FP8+MTP (official, 1-node)
	$(RUN) $(QWEN_RECIPE) $(OVERRIDES)

qwen35: ## Launch Qwen3.6-35B-A3B NVFP4-Fast + MTP, 1M ctx (local, 1-node)
	$(RUN) $(QWEN35_RECIPE) $(OVERRIDES)

hy3: ## Launch Hy3-295B NVFP4-W4A16 + MTP (local, 2-node)
	$(RUN) $(HY3_RECIPE) $(OVERRIDES)

deepseek-dspark: ## Launch DeepSeek-V4-Flash + DSpark drafter (local, 2-node, fp8 KV)
	$(RUN) $(DEEPSEEK_DSPARK_RECIPE) $(OVERRIDES)

## --- dry-run / VRAM fit estimate (no launch) ------------------------------

deepseek-dry: ## Estimate VRAM/context fit for DeepSeek
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES) --dry-run

minimax-dry: ## Estimate VRAM/context fit for MiniMax-M2.7 NVFP4
	$(RUN) $(MINIMAX_RECIPE) $(OVERRIDES) --dry-run

m3-dry: ## Estimate VRAM/context fit for MiniMax-M3 v0 NVFP4 REAP-25
	$(RUN) $(M3_RECIPE) $(OVERRIDES) --dry-run

qwen-dry: ## Estimate VRAM/context fit for Qwen3.6-27B FP8+MTP
	$(RUN) $(QWEN_RECIPE) $(OVERRIDES) --dry-run

qwen35-dry: ## Estimate VRAM/context fit for Qwen3.6-35B-A3B NVFP4-Fast
	$(RUN) $(QWEN35_RECIPE) $(OVERRIDES) --dry-run

hy3-dry: ## Estimate VRAM/context fit for Hy3-295B NVFP4-W4A16
	$(RUN) $(HY3_RECIPE) $(OVERRIDES) --dry-run

deepseek-dspark-dry: ## Estimate VRAM/context fit for DeepSeek-V4-Flash + DSpark
	$(RUN) $(DEEPSEEK_DSPARK_RECIPE) $(OVERRIDES) --dry-run

## --- lifecycle ------------------------------------------------------------

stop: ## Stop all sparkrun workloads on the cluster
	$(SPARKRUN) stop --all --cluster $(CLUSTER)

stop-deepseek: ## Stop just the DeepSeek workload
	$(SPARKRUN) stop $(DEEPSEEK_RECIPE) --cluster $(CLUSTER)

stop-minimax: ## Stop just the MiniMax-M2.7 NVFP4 workload
	$(SPARKRUN) stop $(MINIMAX_RECIPE) --cluster $(CLUSTER)

stop-m3: ## Stop just the MiniMax-M3 v0 NVFP4 REAP-25 workload
	$(SPARKRUN) stop $(M3_RECIPE) --cluster $(CLUSTER)

stop-qwen: ## Stop just the Qwen3.6-27B FP8+MTP workload
	$(SPARKRUN) stop $(QWEN_RECIPE) --cluster $(CLUSTER)

stop-qwen35: ## Stop just the Qwen3.6-35B-A3B NVFP4-Fast workload
	$(SPARKRUN) stop $(QWEN35_RECIPE) --cluster $(CLUSTER)

stop-hy3: ## Stop just the Hy3-295B NVFP4 workload
	$(SPARKRUN) stop $(HY3_RECIPE) --cluster $(CLUSTER)

stop-deepseek-dspark: ## Stop just the DeepSeek-V4-Flash + DSpark workload
	$(SPARKRUN) stop $(DEEPSEEK_DSPARK_RECIPE) --cluster $(CLUSTER)

status: ## Show running sparkrun containers
	$(SPARKRUN) status --cluster $(CLUSTER)

logs: ## Re-attach to running workload logs
	$(SPARKRUN) logs --cluster $(CLUSTER)

list: ## List available recipes
	$(SPARKRUN) list
