"""VLLM-backed LLM serving with OpenAI-compatible API.

Provides production-grade LLM inference with:
- PagedAttention for efficient KV cache management
- Continuous batching for high throughput
- OpenAI-compatible /v1/chat/completions and /v1/completions endpoints
- Quantized inference support (AWQ, GPTQ, FP8)

Usage:
    ludwig serve_llm --model_path path/to/model --port 8000

    # Then call with OpenAI-compatible client:
    curl http://localhost:8000/v1/chat/completions -d '{
        "model": "ludwig-llm",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'

Requires: pip install vllm
"""

import logging
import os

logger = logging.getLogger(__name__)


def create_vllm_app(
    model_path: str,
    model_name: str = "ludwig-llm",
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.9,
    quantization: str | None = None,
    tensor_parallel_size: int = 1,
):
    """Create a FastAPI app with OpenAI-compatible LLM serving backed by vLLM.

    Args:
        model_path: Path to the Ludwig LLM model or HuggingFace model ID.
        model_name: Name to use in the OpenAI API responses.
        max_model_len: Maximum sequence length. None for auto-detect.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
        quantization: Quantization method (awq, gptq, fp8, None).
        tensor_parallel_size: Number of GPUs for tensor parallelism.

    Returns:
        FastAPI application with OpenAI-compatible endpoints.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM is required for LLM serving. Install with: pip install vllm\n"
            "Note: vLLM requires a GPU and CUDA toolkit."
        )

    from fastapi import FastAPI
    from pydantic import BaseModel as PydanticBaseModel

    # Resolve the HuggingFace model path from Ludwig model directory
    # Ludwig stores the fine-tuned model in model/model_weights/
    hf_model_path = model_path
    ludwig_weights = os.path.join(model_path, "model", "model_weights")
    if os.path.isdir(ludwig_weights):
        hf_model_path = ludwig_weights
        logger.info(f"Using Ludwig model weights from {hf_model_path}")

    # Initialize vLLM engine
    engine_kwargs = {
        "model": hf_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
    }
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len
    if quantization is not None:
        engine_kwargs["quantization"] = quantization

    logger.info(f"Initializing vLLM engine with {engine_kwargs}")
    llm = LLM(**engine_kwargs)

    # Pydantic models for OpenAI-compatible API
    class ChatMessage(PydanticBaseModel):
        role: str
        content: str

    class ChatCompletionRequest(PydanticBaseModel):
        model: str = model_name
        messages: list[ChatMessage]
        temperature: float = 1.0
        top_p: float = 1.0
        max_tokens: int = 256
        stop: list[str] | None = None

    class CompletionRequest(PydanticBaseModel):
        model: str = model_name
        prompt: str | list[str]
        temperature: float = 1.0
        top_p: float = 1.0
        max_tokens: int = 256
        stop: list[str] | None = None

    app = FastAPI(title="Ludwig LLM Server (vLLM)", version="0.12.0")

    @app.get("/")
    def health():
        return {"status": "healthy", "backend": "vllm", "model": model_name}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "ludwig"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        # Build prompt from messages
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        prompt = "\n".join(prompt_parts) + "\nAssistant:"

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        return {
            "id": "chatcmpl-ludwig",
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text.strip()},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
                "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids),
            },
        }

    @app.post("/v1/completions")
    def completions(request: CompletionRequest):
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

        outputs = llm.generate(prompts, sampling_params)
        choices = []
        for i, output in enumerate(outputs):
            choices.append(
                {
                    "index": i,
                    "text": output.outputs[0].text,
                    "finish_reason": "stop",
                }
            )

        return {
            "id": "cmpl-ludwig",
            "object": "text_completion",
            "model": model_name,
            "choices": choices,
        }

    return app


def run_vllm_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_name: str = "ludwig-llm",
    **kwargs,
):
    """Run the vLLM-backed LLM serving application."""
    import uvicorn

    app = create_vllm_app(model_path=model_path, model_name=model_name, **kwargs)
    uvicorn.run(app, host=host, port=port)
