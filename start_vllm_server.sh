#!/usr/bin/env bash

set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_path="${HIDREAM_REFINER_MODEL:-${repo_dir}/HiDream-ai/Prompt-Refine}"
served_model_name="${HIDREAM_REFINER_SERVED_NAME:-HiDream-ai/Prompt-Refine}"
host="${HIDREAM_REFINER_HOST:-0.0.0.0}"
port="${HIDREAM_REFINER_PORT:-8000}"
max_model_len="${HIDREAM_REFINER_MAX_MODEL_LEN:-8192}"
tensor_parallel_size="${HIDREAM_REFINER_TENSOR_PARALLEL_SIZE:-1}"
chat_template="${HIDREAM_REFINER_CHAT_TEMPLATE:-${model_path}/tool_chat_template_gemma4.jinja}"

if ! command -v vllm >/dev/null 2>&1; then
    echo "error: vLLM is not installed in this Linux environment." >&2
    echo "Install it with: uv pip install vllm" >&2
    exit 1
fi

if [[ "${model_path}" == */* && ! -f "${model_path}/config.json" ]]; then
    echo "error: no model config found at ${model_path}/config.json" >&2
    echo "Set HIDREAM_REFINER_MODEL to the downloaded Prompt-Refine directory." >&2
    exit 1
fi

if [[ ! -f "${chat_template}" ]]; then
    echo "error: chat template not found: ${chat_template}" >&2
    exit 1
fi

exec vllm serve "${model_path}" \
    --served-model-name "${served_model_name}" \
    --host "${host}" \
    --port "${port}" \
    --tensor-parallel-size "${tensor_parallel_size}" \
    --max-model-len "${max_model_len}" \
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --chat-template "${chat_template}" \
    --mm-processor-kwargs '{"max_soft_tokens": 1120}' \
    "$@"
