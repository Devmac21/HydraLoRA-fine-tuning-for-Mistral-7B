# HydraLoRA Fine-tuning for Mistral-7B

Paper-faithful implementation of **HydraLoRA** (multi-expert LoRA with top-k sparse routing) for parameter-efficient fine-tuning of large language models.

## Key Features

- **8 experts per layer** with learned router
- **Top-2 sparse routing** (paper-faithful) — only 2 experts active per token
- **~0.4% trainable parameters** (~29M for Mistral-7B)
- **Instruction tuning** on UltraChat with proper loss masking
- **Router entropy diagnostics** — expert usage analysis
- **NaN-safe training** — gradient clipping, loss checks

## HydraLoRA vs Standard LoRA

| Aspect | Standard LoRA | HydraLoRA |
|--------|---------------|-----------|
| **Experts** | 1 adapter per layer | 8 experts, top-2 routing |
| **Capacity** | Fixed rank | Multi-expert specialization |
| **Trainable params** | ~0.1–0.2% | ~0.4% (more expressive) |
| **Routing** | N/A | Learned router selects experts |
| **Use case** | General fine-tuning | Domain adaptation, multi-task |

## Results

| Dataset | Steps | Loss (start) | Loss (end) |
|---------|-------|--------------|------------|
| Wikitext-2 | 500 | 5.5 | 1.3 |
| Wikitext-2 | 3000 | 5.7 | 1.6 |
| UltraChat 50k | 1500 | 1.9 | 0.77 |

## Project Structure

```
Xingling/
├── src/
│   └── hydralora.py      # HydraLoRA implementation
├── configs/
│   └── default.yaml     # Training config
├── notebooks/           # Jupyter notebooks
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.hydralora import apply_hydralora, HydraLoRALinear

# Load model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", ...)

# Inject HydraLoRA (top-2 routing)
apply_hydralora(model, top_k=2)

# Freeze base, train only HydraLoRA
for n, p in model.named_parameters():
    p.requires_grad = "hydra" in n.lower()
```

## Citation

If you use this implementation, please cite the HydraLoRA paper and this repository.

## License

MIT
