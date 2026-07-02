# Makefile — convenience wrappers around `sparkrun run` for the recipes Leo runs
# day-to-day on the 2x DGX Spark (GB10) homelab.
#
#   make deepseek                         # launch DeepSeek-V4-Flash (fp8 + MTP)
#   make minimax                          # launch MiniMax-M2.7 NVFP4 (official, default)
#   make minimax-awq                      # launch MiniMax-M2.7 AWQ-4bit (official)
#   make qwen                             # launch Qwen3.6-27B FP8+MTP (official, default)
#   make qwen-fp8                         # launch Qwen3.6-27B FP8, no MTP (official)
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
MINIMAX_NVFP4_RECIPE  := @official/minimax-m2.7-nvfp4-vllm
MINIMAX_AWQ_RECIPE    := @official/minimax-m2.7-awq4-vllm
# `minimax` defaults to the NVFP4 build.
MINIMAX_RECIPE        := $(MINIMAX_NVFP4_RECIPE)
# Qwen3.6-27B — both official builds are FP8 (highest-fidelity quant available);
# the MTP build adds multi-token prediction (same weights/footprint, faster decode).
# These are single-node / TP1 recipes.
QWEN_FP8_RECIPE       := @official/qwen3.6-27b-fp8-vllm
QWEN_MTP_RECIPE       := @official/qwen3.6-27b-fp8-mtp-vllm
# `qwen` defaults to the FP8+MTP build.
QWEN_RECIPE           := $(QWEN_MTP_RECIPE)

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

.PHONY: help deepseek minimax minimax-nvfp4 minimax-awq qwen qwen-fp8 qwen-mtp \
        deepseek-dry minimax-dry minimax-nvfp4-dry minimax-awq-dry \
        qwen-dry qwen-fp8-dry qwen-mtp-dry \
        stop stop-deepseek stop-minimax stop-minimax-nvfp4 stop-minimax-awq \
        stop-qwen stop-qwen-fp8 stop-qwen-mtp \
        status logs list

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- launch ---------------------------------------------------------------

deepseek: ## Launch DeepSeek-V4-Flash (experimental, fp8 + MTP)
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES)

minimax: minimax-nvfp4 ## Launch MiniMax-M2.7 (official, defaults to NVFP4)

minimax-nvfp4: ## Launch MiniMax-M2.7 NVFP4 (official)
	$(RUN) $(MINIMAX_NVFP4_RECIPE) $(OVERRIDES)

minimax-awq: ## Launch MiniMax-M2.7 AWQ-4bit (official)
	$(RUN) $(MINIMAX_AWQ_RECIPE) $(OVERRIDES)

qwen: qwen-mtp ## Launch Qwen3.6-27B (official, defaults to FP8+MTP)

qwen-fp8: ## Launch Qwen3.6-27B FP8, no MTP (official)
	$(RUN) $(QWEN_FP8_RECIPE) $(OVERRIDES)

qwen-mtp: ## Launch Qwen3.6-27B FP8+MTP (official)
	$(RUN) $(QWEN_MTP_RECIPE) $(OVERRIDES)

## --- dry-run / VRAM fit estimate (no launch) ------------------------------

deepseek-dry: ## Estimate VRAM/context fit for DeepSeek
	$(RUN) $(DEEPSEEK_RECIPE) $(OVERRIDES) --dry-run

minimax-dry: minimax-nvfp4-dry ## Estimate VRAM/context fit for MiniMax (NVFP4)

minimax-nvfp4-dry: ## Estimate VRAM/context fit for MiniMax NVFP4
	$(RUN) $(MINIMAX_NVFP4_RECIPE) $(OVERRIDES) --dry-run

minimax-awq-dry: ## Estimate VRAM/context fit for MiniMax AWQ-4bit
	$(RUN) $(MINIMAX_AWQ_RECIPE) $(OVERRIDES) --dry-run

qwen-dry: qwen-mtp-dry ## Estimate VRAM/context fit for Qwen3.6-27B (FP8+MTP)

qwen-fp8-dry: ## Estimate VRAM/context fit for Qwen3.6-27B FP8
	$(RUN) $(QWEN_FP8_RECIPE) $(OVERRIDES) --dry-run

qwen-mtp-dry: ## Estimate VRAM/context fit for Qwen3.6-27B FP8+MTP
	$(RUN) $(QWEN_MTP_RECIPE) $(OVERRIDES) --dry-run

## --- lifecycle ------------------------------------------------------------

stop: ## Stop all sparkrun workloads on the cluster
	$(SPARKRUN) stop --all --cluster $(CLUSTER)

stop-deepseek: ## Stop just the DeepSeek workload
	$(SPARKRUN) stop $(DEEPSEEK_RECIPE) --cluster $(CLUSTER)

stop-minimax: stop-minimax-nvfp4 ## Stop the MiniMax workload (NVFP4)

stop-minimax-nvfp4: ## Stop just the MiniMax NVFP4 workload
	$(SPARKRUN) stop $(MINIMAX_NVFP4_RECIPE) --cluster $(CLUSTER)

stop-minimax-awq: ## Stop just the MiniMax AWQ-4bit workload
	$(SPARKRUN) stop $(MINIMAX_AWQ_RECIPE) --cluster $(CLUSTER)

stop-qwen: stop-qwen-mtp ## Stop the Qwen3.6-27B workload (FP8+MTP)

stop-qwen-fp8: ## Stop just the Qwen3.6-27B FP8 workload
	$(SPARKRUN) stop $(QWEN_FP8_RECIPE) --cluster $(CLUSTER)

stop-qwen-mtp: ## Stop just the Qwen3.6-27B FP8+MTP workload
	$(SPARKRUN) stop $(QWEN_MTP_RECIPE) --cluster $(CLUSTER)

status: ## Show running sparkrun containers
	$(SPARKRUN) status --cluster $(CLUSTER)

logs: ## Re-attach to running workload logs
	$(SPARKRUN) logs --cluster $(CLUSTER)

list: ## List available recipes
	$(SPARKRUN) list
