# Makefile — convenience wrappers around `sparkrun run` for the recipes Leo runs
# day-to-day on the 2x DGX Spark (GB10) homelab.
#
#   make deepseek                         # launch DeepSeek-V4-Flash (fp8 + MTP)
#   make minimax                          # launch MiniMax-M2.7 NVFP4 (official)
#   make deepseek MAX_MODEL_LEN=1000000   # override context length
#   make deepseek-dry                     # VRAM/fit estimate, no launch
#   make stop                             # stop everything on the cluster
#
# Recipes are registry-qualified (@registry/name); resolved via the enabled
# sparkrun registries — no local checkout needed.

SPARKRUN ?= sparkrun
CLUSTER  ?= leo-azl-2node

# Registry-qualified recipe identifiers.
DEEPSEEK_RECIPE := @experimental/deepseek4-flash-fp8-mtp-vllm
MINIMAX_RECIPE  := @official/minimax-m2.7-nvfp4-vllm

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

.PHONY: help deepseek minimax deepseek-dry minimax-dry \
        stop stop-deepseek stop-minimax status logs list

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## --- launch ---------------------------------------------------------------

deepseek: ## Launch DeepSeek-V4-Flash (experimental, fp8 + MTP)
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES)

minimax: ## Launch MiniMax-M2.7 NVFP4 (official)
	$(RUN) $(MINIMAX_RECIPE) $(OVERRIDES)

## --- dry-run / VRAM fit estimate (no launch) ------------------------------

deepseek-dry: ## Estimate VRAM/context fit for DeepSeek
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES) --dry-run

minimax-dry: ## Estimate VRAM/context fit for MiniMax
	$(RUN) $(MINIMAX_RECIPE) $(OVERRIDES) --dry-run

## --- lifecycle ------------------------------------------------------------

stop: ## Stop all sparkrun workloads on the cluster
	$(SPARKRUN) stop --all --cluster $(CLUSTER)

stop-deepseek: ## Stop just the DeepSeek workload
	$(SPARKRUN) stop $(DEEPSEEK_RECIPE) --cluster $(CLUSTER)

stop-minimax: ## Stop just the MiniMax workload
	$(SPARKRUN) stop $(MINIMAX_RECIPE) --cluster $(CLUSTER)

status: ## Show running sparkrun containers
	$(SPARKRUN) status --cluster $(CLUSTER)

logs: ## Re-attach to running workload logs
	$(SPARKRUN) logs --cluster $(CLUSTER)

list: ## List available recipes
	$(SPARKRUN) list
