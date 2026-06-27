# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this repo is

A **local sparkrun recipe registry** — a versioned home for vLLM serving recipes
used in Leo's DGX Spark homelab. There is no application code to build or test;
the "artifacts" are recipe files that `sparkrun` consumes. The repo is itself a
git registry (`.sparkrun/registry.yaml`, registry name `local`), so sparkrun can
add it like any other registry and run recipes by name.

The homelab is **2x DGX Spark (GB10 / Blackwell, `sm_121a`)** — nodes `spark-f31f`
(`169.254.225.22`) and `spark-d306` (`169.254.17.50`) — linked over ConnectX-7
(InfiniBand/RoCE). Recipes are written for this specific two-node topology.

## Layout

```
.sparkrun/registry.yaml   # registry manifest (name: local; points at recipes/ and recipes/mods/)
recipes/
  <name>.yaml             # the recipe sparkrun runs
  <name>.env              # standalone env file mirroring the recipe's env: block
  mods/                   # recipe mods (currently empty)
README.md                 # human-facing usage + per-recipe caveats
```

## Anatomy of a recipe

A `recipes/<name>.yaml` declares: `name`, `model`, `runtime` (e.g. `vllm-distributed`),
node constraints (`min_nodes`, `cluster_only`), `container` image, `metadata`,
`defaults` (CLI-overridable knobs like `tensor_parallel`, `max_model_len`), an
`env:` block, and a `command:` serve template using `{placeholder}` substitution
from `defaults`.

## Conventions to follow

- **Keep `<name>.env` in sync with the YAML `env:` block.** They are intentionally
  duplicated: the YAML drives `sparkrun run`; the `.env` is for raw docker /
  `--env-file` / manual `export` workflows. If you edit one, edit the other.
- **Document provenance and trust.** Recipes here are often third-party/community
  (e.g. from NVIDIA forums) and unverified on this host. Note the source URL,
  whether it's verified, custom container images, and any caveats — in both the
  recipe header comment and README.md.
- **GB10 specifics are load-bearing.** `TORCH_CUDA_ARCH_LIST=12.1a` /
  `FLASHINFER_CUDA_ARCH_LIST=12.1a` and the `NCCL_*` InfiniBand settings are tuned
  for this hardware/interconnect. Don't change them casually.
- **Multi-node coordination is handled by the `vllm-distributed` runtime.** Don't
  hand-add `--nnodes / --node-rank / --master-addr / --master-port / --headless`
  to a recipe `command:` unless a custom entrypoint refuses to start without them;
  sparkrun injects `NODE_RANK` / `MASTER_ADDR` itself.
- When adding a recipe, also add its caveats section to README.md.

## Running (for reference — these run on the Sparks, not here)

```bash
# by file path (no registration needed)
sparkrun run recipes/<name>.yaml --tp 2 --hosts 169.254.225.22,169.254.17.50

# by name (after `sparkrun registry add /home/leo/sparkrun-recipes`)
sparkrun run <name> --tp 2 --cluster <your-2node-cluster>
```
