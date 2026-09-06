#!/usr/bin/env python3
"""Keep sparkrun's cluster control plane off Wi-Fi.

sparkrun derives three things from each host's default-route interface:
the torch master address (scripts/ip_detect.sh), the per-host
GLOO/NCCL/TP socket interface (scripts/ib_detect.sh -> DETECTED_SOCKET_IFNAME),
and — via vLLM's own get_ip(), which also follows the default route unless
VLLM_HOST_IP is set — the address vLLM advertises for its scheduler
broadcast / worker-response message queues (the engine core's
"mq_connect_ip"). On these Sparks the wired 10GbE port has no carrier, so
all three landed on wlP9s9, and the Optus mesh black-holes Spark-to-Spark
Wi-Fi after a roam (ARP for the peer resolves to the mesh satellite's MAC).
That stalled the 2026-09-05 GLM run mid-decode (rank 1's TCPStore reset at
03:04:54 UTC), failed one relaunch in gloo's first barrier, and hung the
next one silently after "reserved 6.0 GiB" with the engine core waiting on
a Wi-Fi ZMQ session that never passed data. NCCL data already rode the CX7
link (NCCL_NET=IB); only the control plane was on Wi-Fi.

This patch makes both detect scripts substitute the first *up* RDMA netdev
that has an IPv4 whenever the default-route interface is wireless, and
teaches sparkrun's env builder to emit VLLM_HOST_IP per host from the
detected management IP. Wired-management clusters see no change (the
substitution never fires, and VLLM_HOST_IP = the mgmt IP vLLM would have
picked anyway).

Idempotent: always rebuilds from the *.orig copies it keeps. `make glm`
runs it; re-run after `sparkrun update` (which reinstalls the package).
"""
import pathlib
import subprocess
import sys

MARK = "# sparkrun-recipes: no-wifi-control-plane"
SPARKRUN_PY = pathlib.Path.home() / ".local/share/uv/tools/sparkrun/bin/python"

pkg = pathlib.Path(
    subprocess.check_output(
        [str(SPARKRUN_PY), "-c", "import sparkrun, pathlib; print(pathlib.Path(sparkrun.__file__).parent)"],
        text=True,
    ).strip()
)
scripts_dir = pkg / "scripts"

RDMA_FALLBACK = f"""{MARK} — a wireless default route is not a usable
# control-plane interface; substitute the first up RDMA netdev that has an IPv4.
if [ -d "/sys/class/net/$DEFAULT_IF/wireless" ]; then
    for _ibnet in /sys/class/infiniband/*/device/net/*; do
        _if=$(basename "$_ibnet")
        if [ "$(cat "/sys/class/net/$_if/operstate" 2>/dev/null)" = "up" ] && ip -4 addr show "$_if" 2>/dev/null | grep -q 'inet '; then
            DEFAULT_IF=$_if
            break
        fi
    done
fi
"""

PATCHES = [
    (
        scripts_dir / "ip_detect.sh",
        "DEFAULT_IF=$(ip route get 8.8.8.8 | grep -oP 'dev \\K\\S+')\n",
        RDMA_FALLBACK,
    ),
    (
        scripts_dir / "ib_detect.sh",
        "DEFAULT_IF=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \\K\\S+' || echo \"eth0\")\n",
        RDMA_FALLBACK,
    ),
    (
        pkg / "orchestration" / "infiniband.py",
        '    if ib_info.get("DETECTED_UCX_LIST"):\n        env["UCX_NET_DEVICES"] = ib_info["DETECTED_UCX_LIST"]\n',
        f"""    {MARK} — vLLM advertises its message-queue
    # address via get_ip() (default route) unless VLLM_HOST_IP is set; pin it per
    # host to the detected management IP (the RDMA netdev's IP when mgmt is wireless).
    if ib_info.get("DETECTED_MGMT_IP"):
        env["VLLM_HOST_IP"] = ib_info["DETECTED_MGMT_IP"]
""",
    ),
]


def patch(path: pathlib.Path, anchor: str, insert: str) -> str:
    orig = path.with_suffix(path.suffix + ".orig")
    if not orig.exists():
        text = path.read_text()
        if MARK in text:
            sys.exit(f"{path.name}: patched but no .orig to rebuild from — restore it by hand")
        orig.write_text(text)
    base = orig.read_text()
    if base.count(anchor) != 1:
        sys.exit(f"{path.name}: anchor not found exactly once in {orig.name} — sparkrun changed; review it")
    new = base.replace(anchor, anchor + insert)
    if path.read_text() == new:
        return f"{path.name}: up to date"
    path.write_text(new)
    return f"{path.name}: patched (original kept at {orig.name})"


for path, anchor, insert in PATCHES:
    print(patch(path, anchor, insert))
print(f"sparkrun package: {pkg}")
