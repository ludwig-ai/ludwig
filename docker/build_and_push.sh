#!/usr/bin/env bash
# Build and push versioned Ludwig Docker images to Docker Hub.
#
# Usage:
#   ./docker/build_and_push.sh <version> [--latest]
#
# Examples:
#   ./docker/build_and_push.sh 0.14.0           # tags: 0.14.0, 0.14
#   ./docker/build_and_push.sh 0.14.0 --latest  # tags: 0.14.0, 0.14, latest
#
# Requires: docker login to ludwigai account already done.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <version> [--latest]"
  echo "  version  full version to build, e.g. 0.14.0"
  echo "  --latest also tag as :latest"
  exit 1
fi

VERSION="$1"
IS_LATEST=false
if [ "${2:-}" = "--latest" ]; then
  IS_LATEST=true
fi

# Derive major.minor tag from full version (e.g. "0.14.0" -> "0.14")
MINOR="${VERSION%.*}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Image variants: "image_name dockerfile_dir"
VARIANTS=(
  "ludwigai/ludwig         docker/ludwig"
  "ludwigai/ludwig-gpu     docker/ludwig-gpu"
  "ludwigai/ludwig-ray     docker/ludwig-ray"
  "ludwigai/ludwig-ray-gpu docker/ludwig-ray-gpu"
)

build_and_push() {
  local image="$1"
  local dockerfile_dir="$2"

  echo ""
  echo "=== Building ${image}:${VERSION} ==="

  local tag_args="-t ${image}:${VERSION} -t ${image}:${MINOR}"
  if [ "${IS_LATEST}" = "true" ]; then
    tag_args="${tag_args} -t ${image}:latest"
  fi

  # shellcheck disable=SC2086
  docker build \
    --build-arg LUDWIG_VERSION="${VERSION}" \
    ${tag_args} \
    -f "${REPO_ROOT}/${dockerfile_dir}/Dockerfile" \
    "${REPO_ROOT}"

  echo "--- Pushing ${image}:${VERSION} ---"
  docker push "${image}:${VERSION}"
  docker push "${image}:${MINOR}"
  if [ "${IS_LATEST}" = "true" ]; then
    docker push "${image}:latest"
  fi
}

for variant_entry in "${VARIANTS[@]}"; do
  read -r image dockerfile_dir <<< "${variant_entry}"
  build_and_push "${image}" "${dockerfile_dir}"
done

echo ""
echo "All images built and pushed successfully."
