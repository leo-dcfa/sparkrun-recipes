---
name: add-recipe
description: >-
  Use when adding a new model or serving recipe to this sparkrun-recipes registry —
  creating a recipes/<name>.yaml, porting a vLLM recipe from an NVIDIA forum thread
  or a GitHub repo, or whenever the user says "add a recipe", "add a model", "extend
  my recipes", or "wire up <model>". This skill runs the FULL add-a-recipe checklist,
  whose most-forgotten step is updating the Makefile with matching targets. Trigger it
  whenever a recipe is being added or a model is being wired into this repo, even if the
  user only mentions the YAML file and never says the word "Makefile".
---

# Add a recipe to sparkrun-recipes

Adding a model to this repo is not just "write a YAML." A recipe is complete only
when four artifacts agree. Steps 1–3 are already in `CLAUDE.md`; **step 4 (the
Makefile) is the one that gets forgotten** — it's why this skill exists.

## The checklist

For a recipe named `<name>` (kebab-case, e.g. `mimo-v2.5-dflash`):

1. **`recipes/<name>.yaml`** — the recipe sparkrun runs. Model, `runtime`,
   `min_nodes`/`cluster_only`, `container`, `metadata` (include `source:` URL and
   trust/verification status — recipes here are usually community/unverified),
   `defaults`, `env:`, and the `command:` serve template. Preserve GB10 load-bearing
   bits (`TORCH_CUDA_ARCH_LIST=12.1a`, `FLASHINFER_CUDA_ARCH_LIST=12.1a`, the `NCCL_*`
   block). Don't hand-add `--nnodes/--node-rank/--master-addr/--headless`; the
   `vllm-distributed` runtime injects those.
2. **`recipes/<name>.env`** — standalone env file mirroring the YAML `env:` block,
   verbatim. If you edit one, edit the other. They are intentionally duplicated.
3. **`README.md`** — add the files to the layout tree and add a `## <name> — caveats`
   section documenting provenance (source URL), the container, verification status,
   and any gotchas.
4. **`Makefile`** — add matching targets (below). **Do this every time.**

After finishing, sanity-check the YAML parses and every `{placeholder}` in the
`command:` resolves to a key in `defaults` (or a top-level field like `model`).

## Step 4: update the Makefile

The Makefile is convenience wrappers around `sparkrun run`. Each model gets a
`_RECIPE` variable plus three targets (`launch`, `-dry`, `stop-`), and each of those
must be added to the `.PHONY` line and mirrored in the header usage comment. The
`help` target greps for `## ` on target lines, so keep the `## ` descriptions.

There are **two flavors** of recipe identifier — pick by whether the recipe is
published to a sparkrun registry yet.

### Flavor A — registry-qualified (published to @official / @experimental / …)

Use when the recipe lives in a published registry. This is the existing pattern
(`deepseek`, `minimax`, `qwen` all use it).

```makefile
# In the "Registry-qualified recipe identifiers" block:
ORNITH_RECIPE := @experimental/ornith-1.0-397b-w4a16-vllm
```

### Flavor B — local recipe, not in any registry yet

Use for a brand-new recipe that only exists as a file in this repo (like the recipes
in `recipes/`). `sparkrun run` accepts a **file path** directly — no registration
needed — so point the variable at the YAML. Mark it clearly as local.

```makefile
# Local recipe (this repo) — runs by file path, no registry needed.
ORNITH_RECIPE := recipes/ornith-1.0-397b.yaml
```

If instead you've run `sparkrun registry add /home/leo/sparkrun-recipes`, you may use
the bare name (`ORNITH_RECIPE := ornith-1.0-397b`) — but the file path always works
and needs no setup, so prefer it for unpublished recipes.

> A recipe can start as Flavor B and graduate to Flavor A once it's published: just
> swap the `_RECIPE` value from the file path to `@registry/id`. The targets don't
> change.

### The targets (identical for both flavors)

After defining `<NAME>_RECIPE`, add these three targets in their respective sections
and register all three in `.PHONY`:

```makefile
## --- launch ---
<name>: ## Launch <Human Name> (<one-line note: quant, nodes, source>)
	$(RUN) $(<NAME>_RECIPE) $(OVERRIDES)

## --- dry-run ---
<name>-dry: ## Estimate VRAM/context fit for <Human Name>
	$(RUN) $(<NAME>_RECIPE) $(OVERRIDES) --dry-run

## --- lifecycle ---
stop-<name>: ## Stop just the <Human Name> workload
	$(SPARKRUN) stop $(<NAME>_RECIPE) --cluster $(CLUSTER)
```

Then, in the same edit:
- Add `<name> <name>-dry stop-<name>` to the `.PHONY:` list.
- Add a usage line to the header comment block near the top (e.g.
  `#   make <name>                           # launch <Human Name>`).

Don't skip `.PHONY` or the header comment — the existing targets all appear in both,
and `make help` depends on the `## ` descriptions.

## Worked example — adding `ornith` (local, Flavor B)

Given `recipes/ornith-1.0-397b.yaml` already exists, the Makefile edits are:

```makefile
# header usage block:
#   make ornith                           # launch Ornith-1.0-397B (INT4, local)

# recipe identifiers:
# Local recipe (this repo) — runs by file path, no registry needed.
ORNITH_RECIPE := recipes/ornith-1.0-397b.yaml

# .PHONY: … ornith ornith-dry stop-ornith …

ornith: ## Launch Ornith-1.0-397B (INT4 W4A16 AutoRound, local, 2-node)
	$(RUN) $(ORNITH_RECIPE) $(OVERRIDES)

ornith-dry: ## Estimate VRAM/context fit for Ornith-1.0-397B
	$(RUN) $(ORNITH_RECIPE) $(OVERRIDES) --dry-run

stop-ornith: ## Stop just the Ornith-1.0-397B workload
	$(SPARKRUN) stop $(ORNITH_RECIPE) --cluster $(CLUSTER)
```

## Done when

All four artifacts exist and agree: `recipes/<name>.yaml`, `recipes/<name>.env` (env
blocks in sync), a README caveats section, and Makefile `<name>` / `<name>-dry` /
`stop-<name>` targets present in both `.PHONY` and the header comment. If the user
only asked for "a recipe" and you stopped after the YAML, you are not done.
