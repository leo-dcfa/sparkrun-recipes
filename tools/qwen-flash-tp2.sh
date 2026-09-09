#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 (nvidia/ModelOpt) on TWO DGX Sparks, TP2 over the RoCE fabric, vLLM. Kai / Tech2Wild 2026-09-05.
#
# LEO: this is upstream's single-spark-vllm-tp1/launch/qwen38fn-nvidia-tp2.sh (tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark
# LEO: @ 6ad1c8f, 2026-09-06) with this pair's deviations, each marked "LEO:" below. Everything else is verbatim upstream.
# LEO:   1. LANE=leo (default): head 10.100.200.2 / worker 10.100.200.1 over the cross-wired CX7 link (head f0 / worker f1);
# LEO:      NCCL_IB_HCA and the GLOO/NCCL/TP socket ifnames are derived per rank from the host IP instead of hard-coded rocep1s0f0.
# LEO:   2. THINKING=1|0 (default 1): server-side enable_thinking, like `make qwen38fn`. Upstream ships thinking OFF.
# LEO:   3. PROFILE=speed|context (default speed): one word for upstream's CONTEXT knob set (PLE_MODE=mmap GRAPHS=piecewise MTP=4 SEQS=8 GMU=0.80 CHUNK=).
# LEO:   4. CHECK=1: preflight only (image, checkpoint, patch files, RDMA device, resolved NICs) and exit 0/1 without launching.
# LEO:   5. config.json of the checkpoint must be HF revision fab0aecb: NVIDIA's fc694b54 (2026-09-05) renamed the MTP experts' quant_algo
# LEO:      FP8_BLOCK_SCALES -> FP8_PB_WO (one line; weights + hf_quant_config.json identical) and this nightly + the overlay then build the
# LEO:      MTP experts unquantized ("mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'"). CHECK=1 warns; the pin is documented in the Makefile.
# LEO: Lives at ~/sparkrun-recipes/tools/qwen-flash-tp2.sh (head) and ~/qwen-flash-tp2.sh (worker, shipped by `make qwen-flash-sync`).
# LEO: Make targets: qwen-flash (THINKING=1), qwen-flash-no-thinking (THINKING=0), qwen-flash-dry (CHECK=1), stop-qwen-flash, logs-qwen-flash.
#
# DEFAULT = SPEED profile (measured 2026-09-05): n-gram table in unified memory (stock loader), CUDA graphs for decode with
# torch.compile off, MTP3, 6 seqs, 4096-token prefill chunks, FP8 KV, gmu 0.70, 262K. 53.7 tok/s median single stream, 97.9 agg at x6,
# KV pool 1.97M. CONTEXT profile (table on disk via our patch, 5.87M pool, 35.8 median): PLE_MODE=mmap GRAPHS=piecewise MTP=4 SEQS=8 GMU=0.80.
# Usage: qwen38fn-nvidia-tp2.sh <0|1>   (run rank 1 = worker FIRST, then rank 0 = head)
# Env knobs: IMAGE, PLE_MODE (mmap = table on disk [CONTEXT] | resident = our patch keeps each rank's slice in memory [SPEED] | none = stock loader, needs GRAPHS=nocompile),
#   GRAPHS (eager|piecewise|full|nocompile|default), LANE (B = Reddie head + Spark4 [default] | A = Bluey head + Asusi), KV_DTYPE, OVERLAYS,
#   PATCH_DIR, GMU, MAXLEN, SEQS, MTP, PORT, MPORT, TOOL_PARSER (qwen3_xml|qwen3_coder), NCCL_CHANNELS (e.g. 8; TJ Klug's 2-Spark recipe pins 8), EXTRA
set -euo pipefail
NODE_RANK="${1:?usage: qwen-flash-tp2.sh <0|1>}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}"
NAME="${NAME:-vllm_qwen38fn}"
MODEL_HOST="${MODEL_HOST:-/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia}"
PATCH_DIR="${PATCH_DIR:-$HOME/patches/qwen4exp-ple-mmap}"
# LEO: PROFILE=context expands to upstream's CONTEXT knob set; explicit knobs on the command line still win.
PROFILE="${PROFILE:-speed}"
case "$PROFILE" in
  speed)   : ;;
  context) PLE_MODE="${PLE_MODE:-mmap}"; GRAPHS="${GRAPHS:-piecewise}"; MTP="${MTP:-4}"; SEQS="${SEQS:-8}"; GMU="${GMU:-0.80}"; CHUNK="${CHUNK-}" ;;
  *) echo "PROFILE must be speed or context" >&2; exit 2 ;;
esac
PLE_MODE="${PLE_MODE:-none}"       # SPEED default (2026-09-05 evening): table in unified memory. CONTEXT: PLE_MODE=mmap
GMU="${GMU:-0.70}"; MAXLEN="${MAXLEN:-262144}"; SEQS="${SEQS:-6}"; MTP="${MTP:-3}"; PORT="${PORT:-8000}"
CHUNK="${CHUNK-4096}"              # --max-num-batched-tokens; the single biggest speed lever measured on GB10 (see README)
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"; GRAPHS="${GRAPHS:-nocompile}"; OVERLAYS="${OVERLAYS:-1}"
THINKING="${THINKING:-1}"          # LEO: 1 = enable_thinking true server-side (upstream: false)
LANE="${LANE:-leo}"                # LEO: upstream default is B
case "$LANE" in
  leo) HEAD_IP="10.100.200.2"; WORKER_IP="10.100.200.1"; MPORT="${MPORT:-29531}"; IB_RANGE="10.100.200.0/24" ;;  # LEO: spark-f31f head, spark-d306 worker
  B) HEAD_IP="192.168.192.2"; WORKER_IP="192.168.192.4"; MPORT="${MPORT:-29531}"; IB_RANGE="192.168.192.0/24" ;;  # Reddie head, Spark4 worker
  A) HEAD_IP="192.168.192.1"; WORKER_IP="192.168.192.3"; MPORT="${MPORT:-29532}"; IB_RANGE="192.168.192.0/24" ;;  # Bluey head, Asusi worker
  *) echo "LANE must be leo, A or B" >&2; exit 2 ;;
esac
case "$NODE_RANK" in
  0) HOST_IP="$HEAD_IP"; HEADLESS="" ;;               # head, serves :PORT
  1) HOST_IP="$WORKER_IP"; HEADLESS="--headless" ;;   # worker
  *) echo "rank must be 0 or 1" >&2; exit 2 ;;
esac
# LEO: resolve the NIC that owns HOST_IP and the RDMA device behind it (cross-wired CX7: head enp1s0f0np0/rocep1s0f0, worker enp1s0f1np1/rocep1s0f1).
IFACE="${IFACE:-$(ip -o -4 addr show 2>/dev/null | awk -v ip="$HOST_IP/" 'index($4, ip)==1 {print $2; exit}')}"
if [ -z "${HCA:-}" ] && [ -n "$IFACE" ]; then
  for d in /sys/class/infiniband/*; do
    [ -d "$d/device/net/$IFACE" ] && HCA="$(basename "$d")" && break
  done
fi
[ -n "$IFACE" ] && [ -n "${HCA:-}" ] || { echo "cannot resolve NIC/HCA for $HOST_IP on $(hostname) (iface='${IFACE:-}' hca='${HCA:-}'); set IFACE= and HCA= explicitly" >&2; exit 4; }
CACHE_HOST="/var/tmp/qwen38fn-vllm-cache"; mkdir -p "$CACHE_HOST"
test -f "$MODEL_HOST/config.json" || { echo "MODEL MISSING at $MODEL_HOST (each rank needs a readable copy: local NVMe, or an NFS mount when PLE_MODE=none)" >&2; exit 3; }
VP=/usr/local/lib/python3.12/dist-packages/vllm
PLE_ENV=(); PLE_MOUNT=()
if [ "$PLE_MODE" = "mmap" ] || [ "$PLE_MODE" = "resident" ] || [ "$PLE_MODE" = "staged" ]; then
  PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-64}")
  # resident: each rank keeps its slice of the FP8 table as a plain GPU tensor behind our gather op
  # (compile stays on, no Inductor copy of the table). TP2: 23.8 GiB per rank, TP4: 11.9 GiB.
  [ "$PLE_MODE" = "resident" ] && PLE_ENV+=(-e QWEN4EXP_PLE_RESIDENT=1)
  PLE_MOUNT=(-v "$PATCH_DIR/ple_layer.py:$VP/models/qwen4_exp/nvidia/ple_layer.py:ro"
             -v "$PATCH_DIR/ple_mmap.py:$VP/models/qwen4_exp/nvidia/ops/ple_mmap.py:ro")
  # staged: rows gathered in the model state's prepare_inputs (before the FULL graph replay) -> decode CUDA graphs with the table on disk
  if [ "$PLE_MODE" = "staged" ]; then
    PLE_ENV+=(-e QWEN4EXP_PLE_STAGED=1)
    PLE_MOUNT+=(-v "$PATCH_DIR/model_state.py:$VP/models/qwen4_exp/nvidia/model_state.py:ro")
  fi
fi
KV_ARGS=(); if [ "$KV_DTYPE" != "auto" ]; then KV_ARGS=(--kv-cache-dtype "$KV_DTYPE"); fi
# DRAFT_VOCAB=65536: reduced-vocabulary drafting for the MTP head (our overlay of vLLM's mtp.py; idea FR-Spec, shown on this model by MiaAI-Lab)
DRAFT_ENV=(); DRAFT_MOUNT=()
if [ -n "${DRAFT_VOCAB:-}" ]; then
  DRAFT_ENV=(-e QWEN4EXP_DRAFT_VOCAB="$DRAFT_VOCAB")
  DRAFT_MOUNT=(-v "$PATCH_DIR/mtp_draft_vocab.py:$VP/models/qwen4_exp/nvidia/mtp.py:ro")
fi
OVERLAY_MOUNT=()
if [ "$OVERLAYS" = "1" ]; then
  OVERLAY_MOUNT=(-v "$PATCH_DIR/upstream-overlays/ops_ple.py:$VP/models/qwen4_exp/nvidia/ops/ple.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/ops_qsa.py:$VP/models/qwen4_exp/nvidia/ops/qsa.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/qsa.py:$VP/models/qwen4_exp/nvidia/qsa.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/platforms_interface.py:$VP/platforms/interface.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/modelopt.py:$VP/model_executor/layers/quantization/modelopt.py:ro")
fi
GRAPH_ARGS=(); GRAPH_MOUNT=()
case "$GRAPHS" in
  eager)     GRAPH_ARGS=(--enforce-eager) ;;
  piecewise) GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
             GRAPH_MOUNT=(-v "$PATCH_DIR/compilation.py:$VP/config/compilation.py:ro") ;;
  full)      GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}')
             GRAPH_MOUNT=(-v "$PATCH_DIR/compilation.py:$VP/config/compilation.py:ro") ;;
  # nocompile: CUDA graphs for decode WITHOUT torch.compile. Needed when the PLE table is resident (PLE_MODE=none):
  # Inductor autotune duplicates the n-gram table during compile (~24 GiB per rank at TP2, ~50 GiB at TP1), which
  # starved and rebooted two Sparks on 2026-09-05. gau-nernst's open vLLM PR #55272 removes compile for that reason.
  nocompile) GRAPH_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
  default)   ;;
  *) echo "GRAPHS must be eager|piecewise|full|nocompile|default" >&2; exit 2 ;;
esac
# MTP_INDEX_SHARE=1 adds index_share_for_mtp_iteration=true (QSA indexer top-k reused across MTP draft steps; knob named by Chuck 208 @CK2084, 2026-09-05)
SPEC=(); if [ "$MTP" != "0" ]; then
  if [ "${MTP_INDEX_SHARE:-0}" = "1" ]; then SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"index_share_for_mtp_iteration\":true}");
  else SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"); fi
fi
# ASYNC_SCHED=1 adds --async-scheduling (also from Chuck 208's list)
ASYNC_ARGS=(); [ "${ASYNC_SCHED:-0}" = "1" ] && ASYNC_ARGS=(--async-scheduling)
# LEO: --max-num-batched-tokens only when CHUNK is non-empty (CHUNK= restores vLLM's default, as upstream's CONTEXT line does)
CHUNK_ARGS=(); [ -n "$CHUNK" ] && CHUNK_ARGS=(--max-num-batched-tokens "$CHUNK")
# LEO: thinking default (upstream hard-codes false)
case "$THINKING" in 1|true|on) THINK_JSON='{"enable_thinking": true}' ;; 0|false|off) THINK_JSON='{"enable_thinking": false}' ;; *) echo "THINKING must be 1 or 0" >&2; exit 2 ;; esac

# LEO: CHECK=1 preflight
if [ "${CHECK:-0}" = "1" ]; then
  rc=0
  echo "[$(hostname)] rank=$NODE_RANK host_ip=$HOST_IP iface=$IFACE hca=$HCA lane=$LANE profile=$PROFILE"
  docker image inspect "$IMAGE" >/dev/null 2>&1 && echo "  ok   image $IMAGE" || { echo "  MISSING image $IMAGE (docker pull it)"; rc=1; }
  n=$(ls "$MODEL_HOST"/model-*-of-*.safetensors 2>/dev/null | wc -l); want=$(ls "$MODEL_HOST"/model-00001-of-*.safetensors 2>/dev/null | sed -E 's/.*-of-0*([0-9]+)\.safetensors/\1/')
  if [ -f "$MODEL_HOST/config.json" ] && [ -n "$want" ] && [ "$n" = "$want" ] && [ -z "$(ls -A "$MODEL_HOST"/.cache/huggingface/download 2>/dev/null | grep incomplete)" ]; then echo "  ok   checkpoint $MODEL_HOST ($n/$want shards, $(du -sh "$MODEL_HOST" 2>/dev/null | cut -f1))"; else echo "  MISSING/partial checkpoint at $MODEL_HOST ($n/${want:-?} shards)"; rc=1; fi
  for f in "${OVERLAY_MOUNT[@]}" "${PLE_MOUNT[@]}" "${DRAFT_MOUNT[@]}" "${GRAPH_MOUNT[@]}"; do
    case "$f" in -v) continue ;; esac; src="${f%%:*}"; [ -f "$src" ] && echo "  ok   $src" || { echo "  MISSING $src"; rc=1; }
  done
  if grep -q FP8_PB_WO "$MODEL_HOST/config.json" 2>/dev/null; then echo "  BAD   config.json is NVIDIA revision fc694b54 (MTP experts FP8_PB_WO): the draft head will not load on this nightly; restore the fab0aecb config.json (see Makefile)"; rc=1; else echo "  ok   config.json MTP quant_algo is FP8_BLOCK_SCALES (revision fab0aecb pin)"; fi
  [ -e /dev/infiniband/rdma_cm ] && echo "  ok   /dev/infiniband" || { echo "  MISSING /dev/infiniband"; rc=1; }
  ip -o -4 addr show dev "$IFACE" >/dev/null 2>&1 && echo "  ok   $IFACE up with $HOST_IP" || { echo "  BAD   $IFACE"; rc=1; }
  docker ps --format '{{.Names}}' | grep -qx "$NAME" && echo "  note $NAME is already running (launch would replace it)"
  echo "  mem  $(grep MemAvailable /proc/meminfo)"
  exit $rc
fi

docker rm -f "$NAME" 2>/dev/null || true
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST:/models/qwen38fn:ro" -v "$CACHE_HOST:/root/.cache" \
  -e VLLM_HOST_IP="$HOST_IP" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="=$HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE="$IB_RANGE" \
  -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e TP_SOCKET_IFNAME="$IFACE" -e MN_IF_NAME="$IFACE" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  ${NCCL_CHANNELS:+-e NCCL_MAX_NCHANNELS=$NCCL_CHANNELS -e NCCL_MIN_NCHANNELS=$NCCL_CHANNELS} \
  "${PLE_ENV[@]}" "${PLE_MOUNT[@]}" "${DRAFT_ENV[@]}" "${DRAFT_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" "${GRAPH_MOUNT[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    /models/qwen38fn --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --quantization modelopt --tensor-parallel-size 2 \
    --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GMU" "${CHUNK_ARGS[@]}" \
    --no-enable-flashinfer-autotune ${PREFIX_CACHE_ARG:---no-enable-prefix-caching} \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_xml}" \
    --default-chat-template-kwargs "$THINK_JSON" \
    "${SPEC[@]}" "${ASYNC_ARGS[@]}" "${GRAPH_ARGS[@]}" "${KV_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" $HEADLESS ${EXTRA:-}
echo "launched $NAME lane=$LANE rank=$NODE_RANK host=$HOST_IP iface=$IFACE hca=$HCA tp=2 profile=$PROFILE ple=$PLE_MODE graphs=$GRAPHS kv=$KV_DTYPE mtp=$MTP gmu=$GMU maxlen=$MAXLEN chunk=${CHUNK:-default} thinking=$THINKING"
sleep 3; docker ps --format "{{.Names}} {{.Status}}" | grep "$NAME" || { echo "$NAME exited"; docker logs "$NAME" 2>&1 | tail -5; exit 1; }
