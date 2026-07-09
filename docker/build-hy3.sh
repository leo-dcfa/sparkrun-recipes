#!/usr/bin/env bash
# Build the local eugr/spark-vllm:hy3-opensource image for the hy3-295b-nvfp4 recipe.
# Run this on each Spark node (or on the head node; sparkrun distributes the image).
set -euo pipefail

cd "$(dirname "$0")"

IMAGE="${IMAGE:-eugr/spark-vllm:hy3-opensource}"
BASE="${BASE:-eugr/spark-vllm:latest}"

echo ">> ensuring base image ${BASE} is present"
docker image inspect "${BASE}" >/dev/null 2>&1 || docker pull "${BASE}"

echo ">> building ${IMAGE} from Dockerfile.hy3"
docker build -f Dockerfile.hy3 -t "${IMAGE}" .

echo ">> done: ${IMAGE}"
docker images | grep -E 'spark-vllm.*hy3-opensource' || true
