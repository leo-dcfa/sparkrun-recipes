#!/usr/bin/env python3
"""Patch sparkrun so 2-node `vllm-ray` recipes get a unique per-node VLLM_HOST_IP.

v2 (2026-07-24): scoped to the vllm_ray runtime ONLY. v1 patched the shared
comm-env builder (orchestration/infiniband.py), which leaked VLLM_HOST_IP into
`vllm-distributed` lanes and broke their c10d rendezvous (worker never joins,
"Timed out ... 1/2 clients joined"). v2 reverts that and instead injects the
value where the ray runtime hands per-host env to the head/worker containers,
and adds it to the Ray carry-over exclusion list.

Why needed at all: vLLM's ray executor requires VLLM_HOST_IP to (a) exist on
the head so the driver's placement pin matches a real Ray node, and (b) be
UNIQUE per node for the worker unique-IP check.

Re-run after every `uv tool upgrade sparkrun`. Idempotent.
"""
import os, sys

BASE = os.path.expanduser(
    "~/.local/share/uv/tools/sparkrun/lib/python3.12/site-packages/sparkrun"
)

# --- 1. Revert v1 (shared infiniband.py) if present -------------------------
ib = os.path.join(BASE, "orchestration", "infiniband.py")
s = open(ib).read()
if "VLLM_HOST_IP" in s:
    anchor = '    env["NODE_IP"] = ib_info.get("DETECTED_MGMT_IP", "")'
    start = s.index(anchor) + len(anchor)
    end = s.index('    env["VLLM_HOST_IP"] = ib_info.get("DETECTED_MGMT_IP", "")', start)
    end += len('    env["VLLM_HOST_IP"] = ib_info.get("DETECTED_MGMT_IP", "")')
    s = s[: s.index(anchor)] + anchor + s[end:]
    open(ib, "w").write(s)
    print("infiniband.py: v1 patch REVERTED")
else:
    print("infiniband.py: clean")

# --- 2. Apply v2 (vllm_ray.py only) -----------------------------------------
vr = os.path.join(BASE, "runtimes", "vllm_ray.py")
s = open(vr).read()
if "VLLM_HOST_IP" in s:
    print("vllm_ray.py: already patched")
    sys.exit(0)

INJECT = (
    '\n        # LOCAL PATCH v2 (leo, sparkrun-recipes/docker/patch-sparkrun.py):'
    "\n        # vLLM's ray executor needs a UNIQUE per-node VLLM_HOST_IP (driver"
    "\n        # placement pin + worker unique-IP check). Scoped to vllm-ray only —"
    "\n        # setting it in the shared comm env breaks vllm-distributed rendezvous."
)

a1 = "        head_nccl_env = comm_env.get_env(ctx.head_host) if comm_env else None"
r1 = a1 + INJECT + (
    '\n        if head_nccl_env and head_nccl_env.get("NODE_IP"):'
    '\n            head_nccl_env = {**head_nccl_env, "VLLM_HOST_IP": head_nccl_env["NODE_IP"]}'
)

a2 = "                    _whost_env = comm_env.get_env(_whost) if comm_env else None"
r2 = a2 + (
    '\n                    if _whost_env and _whost_env.get("NODE_IP"):  # LOCAL PATCH v2'
    '\n                        _whost_env = {**_whost_env, "VLLM_HOST_IP": _whost_env["NODE_IP"]}'
)

a3 = "        exclude_vars = sorted(comm_env.per_host_keys())"
r3 = '        exclude_vars = sorted(set(comm_env.per_host_keys()) | {"VLLM_HOST_IP"})  # LOCAL PATCH v2'

for a, r in ((a1, r1), (a2, r2), (a3, r3)):
    if a not in s:
        print(f"ANCHOR NOT FOUND — sparkrun layout changed, patch manually:\n{a}")
        sys.exit(1)
    s = s.replace(a, r, 1)

open(vr, "w").write(s)
print("vllm_ray.py: v2 patch APPLIED")
