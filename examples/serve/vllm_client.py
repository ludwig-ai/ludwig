"""Ludwig vLLM server client.

Demonstrates how to call a running Ludwig vLLM server via its REST API,
covering the /predict and /batch_predict endpoints and reporting throughput.

Usage:
    # Start a Ludwig vLLM server first:
    #   ludwig serve --model_path=/path/to/llm_model --backend vllm --num_gpus 1
    python vllm_client.py
"""

import json
import sys
import time

import requests

SERVER_URL = "http://localhost:8000"

PREDICT_URL = f"{SERVER_URL}/predict"
BATCH_PREDICT_URL = f"{SERVER_URL}/batch_predict"
OPENAI_URL = f"{SERVER_URL}/v1/completions"

# ---------------------------------------------------------------------------
# Sample data — adjust field names to match your trained model's input features
# ---------------------------------------------------------------------------
SAMPLE_TEXT = "Explain the concept of retrieval-augmented generation in one sentence."

BATCH_SAMPLES = [
    "What is Ludwig?",
    "Summarise the key benefits of vLLM.",
    "Describe how PagedAttention works.",
    "What is tensor parallelism?",
    "How does continuous batching improve GPU utilisation?",
]


def _check_server() -> bool:
    """Return True when the server is reachable."""
    try:
        resp = requests.get(f"{SERVER_URL}/", timeout=5)
        return resp.status_code < 500
    except requests.exceptions.ConnectionError:
        return False


# ---------------------------------------------------------------------------
# /predict  (single example)
# ---------------------------------------------------------------------------
def predict_single(text: str) -> dict:
    """Send a single text to the /predict endpoint and return the response."""
    t0 = time.perf_counter()
    response = requests.post(PREDICT_URL, data={"text": text}, timeout=120)
    elapsed = time.perf_counter() - t0

    response.raise_for_status()
    result = response.json()
    print(f"[predict] latency: {elapsed * 1000:.1f} ms")
    print(f"[predict] response: {json.dumps(result, indent=2)}")
    return result


# ---------------------------------------------------------------------------
# /batch_predict  (multiple examples)
# ---------------------------------------------------------------------------
def predict_batch(texts: list[str]) -> dict:
    """Send a batch of texts to the /batch_predict endpoint."""
    dataset = json.dumps({"columns": ["text"], "data": [[t] for t in texts]})

    t0 = time.perf_counter()
    response = requests.post(BATCH_PREDICT_URL, data={"dataset": dataset}, timeout=300)
    elapsed = time.perf_counter() - t0

    response.raise_for_status()
    result = response.json()

    n = len(texts)
    throughput = n / elapsed
    print(f"[batch_predict] {n} examples in {elapsed:.2f} s — {throughput:.1f} examples/s")
    print(f"[batch_predict] columns: {result.get('columns')}")
    for i, row in enumerate(result.get("data", [])[:3]):
        print(f"  [{i}] {row}")
    if n > 3:
        print(f"  ... ({n - 3} more)")
    return result


# ---------------------------------------------------------------------------
# /v1/completions  (OpenAI-compatible, exposed by vLLM)
# ---------------------------------------------------------------------------
def openai_compat_request(prompt: str) -> dict:
    """Call the OpenAI-compatible completions endpoint exposed by vLLM."""
    payload = {
        "model": "ludwig-model",  # vLLM uses this as a placeholder name
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    response = requests.post(OPENAI_URL, json=payload, timeout=120)
    elapsed = time.perf_counter() - t0

    if response.status_code == 404:
        print("[openai] /v1/completions not available on this server (requires --enable-openai-api flag)")
        return {}

    response.raise_for_status()
    result = response.json()
    print(f"[openai] latency: {elapsed * 1000:.1f} ms")
    choices = result.get("choices", [])
    if choices:
        print(f"[openai] completion: {choices[0].get('text', '').strip()}")
    return result


# ---------------------------------------------------------------------------
# Throughput benchmark: compare repeated single calls vs one batch call
# ---------------------------------------------------------------------------
def throughput_benchmark(texts: list[str], warmup: int = 1) -> None:
    """Compare per-request latency (sequential) vs batch throughput."""
    print("\n--- Throughput benchmark ---")

    # Warmup
    for _ in range(warmup):
        requests.post(PREDICT_URL, data={"text": texts[0]}, timeout=120)

    # Sequential single requests
    t0 = time.perf_counter()
    for text in texts:
        requests.post(PREDICT_URL, data={"text": text}, timeout=120)
    seq_elapsed = time.perf_counter() - t0
    seq_tps = len(texts) / seq_elapsed
    print(f"Sequential /predict:  {seq_elapsed:.2f} s total — {seq_tps:.1f} examples/s")

    # Single batch request
    dataset = json.dumps({"columns": ["text"], "data": [[t] for t in texts]})
    t0 = time.perf_counter()
    requests.post(BATCH_PREDICT_URL, data={"dataset": dataset}, timeout=300)
    batch_elapsed = time.perf_counter() - t0
    batch_tps = len(texts) / batch_elapsed
    print(f"Batch   /batch_predict: {batch_elapsed:.2f} s total — {batch_tps:.1f} examples/s")
    print(f"Batch speedup: {seq_tps / max(batch_tps, 1e-9):.2f}x")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if not _check_server():
        print(
            f"Cannot reach Ludwig server at {SERVER_URL}.\n"
            "Start it with:\n"
            "  ludwig serve --model_path=/path/to/llm_model --backend vllm --num_gpus 1",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=== Single prediction ===")
    predict_single(SAMPLE_TEXT)

    print("\n=== Batch prediction ===")
    predict_batch(BATCH_SAMPLES)

    print("\n=== OpenAI-compatible endpoint ===")
    openai_compat_request(SAMPLE_TEXT)

    print("\n=== Throughput benchmark ===")
    throughput_benchmark(BATCH_SAMPLES)


if __name__ == "__main__":
    main()
