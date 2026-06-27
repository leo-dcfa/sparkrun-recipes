# Leo's local sparkrun recipes

Local recipe registry for the DGX Spark homelab (2x GB10, nodes `spark-f31f`
`169.254.225.22` + `spark-d306` `169.254.17.50`, linked over ConnectX-7).

```
sparkrun-recipes/
├── .sparkrun/registry.yaml          # registry manifest (name: local)
└── recipes/
    ├── deepseek-v4-flash.yaml       # the recipe
    ├── deepseek-v4-flash.env        # standalone env file (mirrors the recipe env: block)
    └── mods/                        # recipe mods (empty for now)
```

## Running

By file path (works immediately, no registration):

```bash
sparkrun run sparkrun-recipes/recipes/deepseek-v4-flash.yaml \
  --tp 2 --hosts 169.254.225.22,169.254.17.50
```

By name (after registering this dir as a registry — see below):

```bash
sparkrun run deepseek-v4-flash --tp 2 --cluster <your-2node-cluster>
```

## Registering as a local registry

This folder is a git repo with a `.sparkrun/registry.yaml` manifest, so sparkrun
can treat it like any other registry (git can clone from a local path):

```bash
sparkrun registry add /home/leo/sparkrun-recipes
sparkrun registry update
sparkrun list | grep -i deepseek
```

To remove it later: `sparkrun registry remove local`.

## deepseek-v4-flash — caveats

This is the community **"Aiden" recipe** from the NVIDIA developer forums
([thread](https://forums.developer.nvidia.com/t/deepseek-v4-flash-aiden-recipe-from-reddit-1m-token-session-operational-cuda-12-1-tailored-for-dgx-spark-gb10/372268)).
It is **unverified on this host**. Things to know before relying on it:

- **2 Sparks required** (`cluster_only`, `min_nodes: 2`). One GB10 per Spark →
  `tensor_parallel: 2` needs both nodes. It cannot run solo.
- **Custom image** `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` (~large,
  first pull/sync is slow) with a baked-in `dsv4-vllm-entrypoint` and DeepSeek-V4
  patches. It is a third-party image — inspect/trust accordingly.
- **CUDA arch is pinned to `12.1a`** (`TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST`)
  for GB10 Blackwell. Don't change unless you know your arch.
- **Multi-node coordination flags omitted.** The forum command passes
  `--nnodes 2 --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port 25000
  ${HEADLESS:+--headless}` because it was written for raw docker-compose. sparkrun's
  `vllm-distributed` runtime does this coordination itself, so they're left out of the
  recipe command. **If the entrypoint refuses to start without them**, re-add that line
  to the `command:` block in `deepseek-v4-flash.yaml` (sparkrun exposes `NODE_RANK` /
  `MASTER_ADDR` in the container env; the env file documents the values).
- **Performance baseline from the author:** ~30 tok/s decode at 980K context, with a
  ~16-minute time-to-first-token. The 1M-token window is real but very slow to fill.
- Keep `deepseek-v4-flash.env` in sync with the `env:` block in the YAML if you edit either.
