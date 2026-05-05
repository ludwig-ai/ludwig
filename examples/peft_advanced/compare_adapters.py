"""Compare trainable parameter counts across PEFT adapters on a tiny GPT-2 model."""

from __future__ import annotations

from transformers import AutoModelForCausalLM

ADAPTERS = [
    ("lora (r=8)", {"type": "lora", "r": 8, "alpha": 16}),
    ("lora+pissa (r=8)", {"type": "lora", "r": 8, "alpha": 16, "init_lora_weights": "pissa"}),
    ("lora+corda (r=8)", {"type": "lora", "r": 8, "alpha": 16, "init_lora_weights": "corda"}),
    ("lora+rslora (r=8)", {"type": "lora", "r": 8, "alpha": 16, "use_rslora": True}),
    ("lora+dora (r=8)", {"type": "lora", "r": 8, "alpha": 16, "use_dora": True}),
    ("tinylora (r=2, u=64)", {"type": "tinylora", "r": 2, "u": 64}),
    ("tinylora (r=2, u=13)", {"type": "tinylora", "r": 2, "u": 13}),
    # OFT/HRA/VBLoRA require nn.Linear layers; not compatible with GPT2's Conv1D.
    # They work correctly on Llama, Mistral, Falcon, etc.
    # ("oft (block=32)", {"type": "oft", "oft_block_size": 32}),
    # ("hra (r=8)", {"type": "hra", "r": 8}),
    # ("vblora (r=4)", {"type": "vblora", "r": 4, "num_vectors": 256, "vector_length": 768, "topk": 2}),
    ("ln_tuning", {"type": "ln_tuning"}),
    ("ia3", {"type": "ia3"}),
    ("vera (r=256)", {"type": "vera", "r": 256}),
    ("adalora (r=8)", {"type": "adalora", "r": 8, "target_r": 4, "init_r": 12, "total_step": 100}),
]

BASE_MODEL = "sshleifer/tiny-gpt2"


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model):
    return sum(p.numel() for p in model.parameters())


def main():
    from peft import get_peft_model

    from ludwig.schema.llms.peft import adapter_registry

    print(f"Base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    total = count_total(base)
    print(f"Total parameters: {total:,}\n")
    print(f"{'Adapter':<30} {'Trainable':>12} {'% of total':>12}")
    print("-" * 58)

    for name, config_dict in ADAPTERS:
        try:
            adapter_type = config_dict["type"]
            if adapter_type not in adapter_registry:
                print(f"{name:<30} {'N/A (not registered)':>25}")
                continue

            cls = adapter_registry[adapter_type]
            inst = cls.model_validate(config_dict)
            peft_cfg = inst.to_config(task_type="CAUSAL_LM")

            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
            peft_model = get_peft_model(model, peft_cfg)
            trainable = count_trainable(peft_model)
            pct = 100.0 * trainable / total
            print(f"{name:<30} {trainable:>12,} {pct:>11.4f}%")
        except Exception as e:
            print(f"{name:<30} {'ERROR: ' + str(e)[:40]:>50}")

    print()
    print("Full fine-tuning would train all", f"{total:,}", "parameters (100%)")


if __name__ == "__main__":
    main()
