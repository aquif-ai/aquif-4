# aquif-4

**aquif-4-Exp** is an experimental research preview of the upcoming aquif-4 family of models. It represents a significant architectural departure from the aquif-3.5 series, introducing hybrid attention mechanisms and advanced mixture-of-experts configurations. This model is not positioned as a direct successor to aquif-3.5, but rather as a proof-of-concept for next-generation innovations in the aquif model family.

**Release Date:** October 15, 2025

## News
- [10.18.2025] 🔥 vLLM support for Aquif4Linear released
- [10.17.2025] 🔥 GitHub repo for aquif-4 created [here](https://github.com/aquif-ai/aquif-4)
- [10.15.2025] 🔥 aquif-4-Exp (16B A3B) released

## Model Overview

| Attribute | Value |
|-----------|-------|
| Total Parameters | 16.45B |
| Active Parameters | 3.2B |
| Activation Ratio | 1:16 |
| Expert Count | 256 |
| Experts per Token | 16 |
| Attention Type | Hybrid (Softmax + Linear) |
| Context Window | 128K (expandable to 512K via YaRN) |
| Is Reasoning Model? | ✅ |
| Model Type | Mixture-of-Experts (MoE) |

## Key Features

### Hybrid Attention Mechanism

aquif-4-Exp is the first aquif model to implement a hybrid attention architecture combining:

- **Softmax Attention**: Applied at strategic layers for precise token interactions and complex reasoning patterns
- **Linear Attention**: Leverages Lightning Attention-2 (https://arxiv.org/abs/2401.04658) for efficient long-context processing

This combination enables efficient processing of extended sequences while maintaining the reasoning capabilities necessary for complex problem-solving tasks.

### Mixture-of-Experts Architecture

- **256 total experts** with a **16 expert activation** strategy
- **1:16 activation ratio** provides exceptional parameter efficiency
- Only 3.2B parameters are active during inference, enabling deployment on resource-constrained hardware while maintaining performance comparable to much larger dense models
- Expert routing is optimized for both training stability and inference efficiency

### Extended Context Support

- **128K native context window** for long-document processing
- **Expandable to 512K tokens** using YaRN (Yet another RoPE extensioN) without full retraining
- Efficient handling of multi-document scenarios and extensive code repositories

## Architecture Details

The aquif-4-Exp implementation builds upon the Aquif4Linear architecture, featuring:

- **Rotary Position Embeddings (RoPE)** with optional scaling via YaRN
- **Group-normalized RMSNorm** for stable layer normalization across attention heads
- **Efficient KV caching** for accelerated inference
- **Optimized flash-linear-attention operators** from the FLA library

## Performance Characteristics

As an experimental research model, aquif-4-Exp demonstrates:

- **Reasoning-focused performance**: Optimized for complex problem-solving and multi-step inference
- **Efficiency at scale**: 3.2B active parameters achieve competitive performance with larger models
- **Multilingual support**: Native support for English, German, Italian, Portuguese, French, Hindi, Spanish, Thai, Chinese, and Japanese
- **Long-context understanding**: Maintains coherence and reasoning quality across extended sequences

## Evaluation

### Speed
<div style="display: flex; gap: 20px; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 300px;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/6747320df82ae35f0327cdd3/jKGFTvQYefJ63LS5r-vYS.png" style="width: 100%; max-width: 400px;" alt="Speed evaluation">
    <small><b>Figure 1:</b> aquif-4-Exp and aquif-3.5-Think on Context Length x Normalized Prefill Throughput</small>
  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/6747320df82ae35f0327cdd3/DZ1UK_9I60S07itzP6ui6.png" style="width: 100%; max-width: 400px;" alt="Speed evaluation continued">
    <small><b>Figure 2:</b> aquif-4-Exp and aquif-3.5-Think on Generation Length x Normalized Decode Throughput</small>
  </div>
</div>

### Performance
<img src="https://cdn-uploads.huggingface.co/production/uploads/6747320df82ae35f0327cdd3/jJG9ghiBThsEyGnOMIFM_.png" style="max-width: 768px; width: 100%;" alt="Performance evaluation">
<small><b>Figure 3:</b> aquif-4-Exp and others evaluated on MMLU-Pro, AIME 2025, LiveCodeBench and GPQA Diamond (Chart).</small>

| Metric | aquif-4-Exp (16B A3.2B) | aquif-3.5-Think (8.2B) | Qwen3-VL-Thinking-2510 (8.8B) | Ring-mini-2.0 (16.3B A1.4B) | gpt-oss (21B A3.6B) |
|--------|-------------------------|------------------------|--------------------------------|------------------------------|---------------------|
| MMLU-Pro | 76.9 | 78.1 | 77.3 | 66.8 | 71.5 |
| AIME 2025 | 82.3 | 81.4 | 80.3 | 74.1 | 72.1 |
| LiveCodeBench | 65.7 | 61.5 | 58.6 | 62.6 | 54.9 |
| GPQA Diamond | 70.1 | 66.8 | 69.9 | 68.2 | 66.0 |
| Average | 73.8 | 72.0 | 71.5 | 67.9 | 66.1 |

<small><b>Figure 4:</b> aquif-4-Exp and others evaluated on MMLU-Pro, AIME 2025, LiveCodeBench and GPQA Diamond (Table).</small>

## Installation

### Requirements

```bash
pip install flash-linear-attention==0.3.2

# For inference with HuggingFace Transformers
pip install transformers==4.56.1

# For inference with vLLM
pip install torch==2.7.0 torchvision==0.22.0
pip install https://github.com/aquif-ai/aquif-4/raw/refs/heads/main/inference/vllm0.8.5-cuda12.8-gcc10.2.1-cp310-cp310-linux_x86_64.whl --no-deps --force-reinstall

```

**Note:** aquif-4-Exp is currently supported only through the Hugging Face Transformers library. Support for llama.cpp, vLLM, and SGLang is coming soon and will be available with the full aquif-4 family release.

## Usage

### 🤗 Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "aquif-ai/aquif-4-Exp"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts = [
    "Hello World"
]

input_texts = []
for prompt in prompts:
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    input_texts.append(text)

print(input_texts)

model_inputs = tokenizer(
    input_texts,
    return_tensors="pt",
    return_token_type_ids=False,
    padding=True,
    padding_side='left'
).to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192,
    do_sample=False,
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print("*" * 30)
print(responses)
print("*" * 30)
```

### ⚙️ vLLM

#### Offline inference
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("aquif-ai/aquif-4-Exp")

sampling_params = SamplingParams(temperature=0.6, top_p=1.0, max_tokens=8192)

llm = LLM(model="aquif-ai/aquif-4-Exp", dtype='bfloat16', enable_prefix_caching=False)
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
outputs = llm.generate([text], sampling_params)
```

#### Online inference
```bash
vllm serve aquif-ai/aquif-4-Exp \
              --tensor-parallel-size 1 \
              --gpu-memory-utilization 0.90 \
              --no-enable-prefix-caching

```

### Enabling Extended Context with YaRN

To use the model with context windows beyond the default 128K tokens, you can configure YaRN scaling in the model's configuration before loading:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_name = "aquif-ai/aquif-4-Exp"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

# Configure YaRN for 512K context
config.rope_scaling = {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 131072,
}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Inference Framework Support

- **Transformers (Native)**: ✅ Full support
- **llama.cpp**: ❌ Not supported
- **vLLM**: ✅ Support through wheel
- **SGLang**: ⏳ Coming soon

Framework support will be expanded with the full aquif-4 family release.

## Usage Recommendations

aquif-4-Exp is designed for:

- **Research applications** exploring hybrid attention mechanisms and MoE architectures
- **Reasoning-heavy tasks** requiring interpretable chain-of-thought outputs
- **Long-context processing** for documents, code analysis, and multi-turn conversations
- **Efficiency-critical deployments** where parameter count matters as much as performance

## Limitations and Considerations

- **Experimental status**: This is a research preview. Stability and performance may evolve with updates
- **CoT overhead**: Chain-of-thought reasoning increases generation latency compared to direct answering
- **Hardware requirements**: Despite 3.2B active parameters, peak memory usage during inference can be higher due to expert loading
- **Not a full successor**: aquif-4-Exp does not replace aquif-3.5 for production use cases; it represents architectural exploration
- **Transformers-only**: Currently requires Hugging Face Transformers; integration with other frameworks is forthcoming

## Technical Specifications

- **Attention Implementation**: Hybrid softmax + linear (Lightning Attention-2)
- **Precision Support**: BF16, FP16
- **Position Encoding**: RoPE with YaRN scaling capability
- **Training Data**: Multilingual corpus spanning 10+ languages
- **Model Family**: First of the upcoming aquif-4 experimental series

## Inference Optimization

For optimal performance:

- Use flash-attention-2 or SDPA for softmax attention layers when available
- Consider YaRN configuration for context windows beyond 128K
- Monitor VRAM usage with full expert loading enabled
- Leverage KV caching for multi-turn conversations
- Ensure `trust_remote_code=True` is set when loading from Hugging Face Hub

## About aquif-4 Full Release

aquif-4-Exp represents the first experimental release in the aquif-4 family exploration. The full aquif-4 release will not be a single model, but rather a comprehensive family of models with varying architectures, sizes, and specializations, all leveraging the innovations demonstrated in this experimental preview.

## Acknowledgements

- **aquif AI Research Team**: Architecture design and optimization
- **EleutherAI & HuggingFace**: GPT-NeoX and modeling foundations
- **Flash Linear Attention Project**: FLA library for efficient kernel implementations
- **Lightning Attention Authors**: Attention mechanism research

## License

This project is released under the Apache 2.0 License.

---

**Note**: aquif-4-Exp is a research release. For production applications, please refer to the aquif-3.5 model series. Feedback and findings from this experimental release will inform the development of the full aquif-4 family.

*Made in 🇧🇷*

© 2025 aquif AI. All rights reserved.
