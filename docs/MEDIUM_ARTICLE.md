# A Practical Guide to Tuning AI Agents with Reinforcement Learning

*Using Strands SDK, strands-vllm, and Agent Lightning to build a complete agent tuning pipeline*

---

## Why Agent Tuning Matters

Large Language Models are powerful, but they're trained on general data. **Agent tuning** lets you adapt them to your specific tasks — teaching them to use tools, follow protocols, or solve domain-specific problems.

This guide walks through building a complete RL pipeline for agent tuning, focusing on:
- **The architecture** — how the pieces fit together
- **TITO (Token-In-Token-Out)** — why it's critical for RL
- **Strands SDK** — building clean, trainable agents
- **The training loop** — from generation to gradient updates

---

## The Core Problem: Retokenization Drift

Before diving into code, let's understand **why agent tuning is tricky**.

When you train with RL, the model generates tokens. But if you convert those tokens to text and back for training, you might get **different tokens**:

```
Model generates: tokens [100, 200, 300] → text "42"
Re-tokenize for training: text "42" → tokens [100, 201, 300]  ← DIFFERENT!
```

This is called **retokenization drift**. Your gradients update the wrong tokens, causing unstable or ineffective training.

### The Solution: TITO

**Token-In-Token-Out (TITO)** preserves the exact token IDs from generation through training. No conversion, no drift.

This is where **strands-vllm** comes in.

---

## The Stack: Strands + strands-vllm + Agent Lightning

### Layer 1: Strands Agent SDK

[Strands](https://github.com/strands-agents/sdk-python) provides a clean abstraction for building AI agents:

```python
from strands import Agent

agent = Agent(
    model=my_model,
    system_prompt="You are a helpful assistant.",
    tools=[my_tool],  # Optional: add tool calling
)

result = agent("What is 15 + 27?")
```

**Why Strands for tuning?**
- Clean separation of agent logic from model details
- Built-in tool calling support
- Callback system for capturing training data

### Layer 2: strands-vllm (The TITO Layer)

[strands-vllm](https://github.com/strands-agents/strands-vllm) connects Strands to vLLM with **TITO support**:

```python
from strands import Agent
from strands_vllm import VLLMModel, VLLMTokenRecorder

# Create model with TITO enabled
model = VLLMModel(
    base_url="http://localhost:8000/v1",
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    return_token_ids=True,  # ← CRITICAL: Request token IDs
)

# Capture token IDs during generation
recorder = VLLMTokenRecorder()

# Create agent with recorder
agent = Agent(
    model=model,
    system_prompt="Solve math problems. End with 'Answer: [number]'",
    callback_handler=recorder,  # ← Hooks into Strands event system
)

# Run agent
result = agent("A store has 45 apples. If 12 are sold, how many remain?")

# Access captured tokens for training
print(f"Prompt tokens: {recorder.prompt_token_ids}")
print(f"Response tokens: {recorder.token_ids}")
```

**Key point:** `VLLMTokenRecorder` captures the exact tokens from vLLM's response. These are the tokens you'll use for gradient updates.

### Layer 3: Agent Lightning (The Training Infrastructure)

[Agent Lightning](https://microsoft.github.io/agent-lightning/) provides the infrastructure for scaling RL training:

```
Algorithm ←→ LightningStore ←→ Runner
   │              │               │
   │              │               ├── Executes rollouts
   │              │               ├── Captures spans (with token IDs!)
   │              │               └── Reports rewards
   │              │
   │              └── Central database for tasks, results, resources
   │
   └── Enqueues tasks, processes results, updates model
```

For this guide, we'll use a simplified local setup, but the same patterns scale to distributed training.

---

## The Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT TUNING PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────────┐  │
│  │   Strands    │─────▶│ strands-vllm │─────▶│  vLLM Server  │  │
│  │    Agent     │      │    (TITO)    │      │    (GPU)      │  │
│  └──────────────┘      └──────────────┘      └───────────────┘  │
│         │                     │                                  │
│         │                     │                                  │
│         ▼                     ▼                                  │
│   Agent Output          Token IDs                                │
│   (text answer)         (for training)                           │
│         │                     │                                  │
│         └──────────┬──────────┘                                  │
│                    ▼                                             │
│         ┌──────────────────────────────────────┐                │
│         │          TRAINING LOOP               │                │
│         │                                      │                │
│         │  1. Generate response (with TITO)   │                │
│         │  2. Compute reward (task-specific)  │                │
│         │  3. Update model (policy gradient)  │                │
│         │                                      │                │
│         └──────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Building the Training Loop

### Step 1: Data Collection with TITO

The first step is collecting (prompt, response, token_ids, reward) tuples:

```python
from strands import Agent
from strands_vllm import VLLMModel, VLLMTokenRecorder

def collect_rollout(problem: dict) -> dict:
    """Execute one rollout and collect TITO data."""
    
    # 1. Create model with TITO
    model = VLLMModel(
        base_url="http://localhost:8000/v1",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        return_token_ids=True,
    )
    
    # 2. Set up token capture
    recorder = VLLMTokenRecorder()
    agent = Agent(
        model=model,
        system_prompt="Solve math problems. End with 'Answer: [number]'",
        callback_handler=recorder,
    )
    
    # 3. Run agent
    result = agent(problem["question"])
    response_text = result.message.content[0].text
    
    # 4. Compute reward (task-specific)
    predicted = extract_answer(response_text)
    reward = 1.0 if predicted == problem["expected"] else 0.0
    
    # 5. Return TITO data
    return {
        "prompt": problem["question"],
        "response": response_text,
        "prompt_token_ids": recorder.prompt_token_ids,
        "response_token_ids": recorder.token_ids,
        "reward": reward,
    }
```

**Why this matters:** The `response_token_ids` are the exact tokens the model generated. When we compute gradients, we'll use these — not re-tokenized text.

### Step 2: The Reward Function

The reward function defines what "good" means for your task:

```python
def compute_reward(response: str, expected: float) -> float:
    """
    Task-specific reward function.
    
    For math problems: 1.0 if correct, 0.0 if wrong.
    For other tasks, you might use:
    - Partial credit based on similarity
    - Tool calling success rate
    - External validators
    """
    predicted = extract_answer(response)
    if predicted is None:
        return 0.0
    return 1.0 if abs(predicted - expected) < 1e-6 else 0.0
```

### Step 3: Policy Gradient Update

Now the core RL step — updating the model based on rewards:

```python
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model with LoRA for efficient training
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
for rollout in collected_data:
    if rollout["reward"] > 0:  # Only learn from successes
        # Use the EXACT token IDs from generation
        input_ids = torch.tensor([rollout["response_token_ids"]])
        
        # Compute loss (negative log likelihood)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Key insight:** We use the original `response_token_ids` from TITO — no re-tokenization needed.

---

## The Sampling vs. Greedy Challenge

A critical detail for practical RL training:

### The Problem

Standard REINFORCE samples diverse outputs during training:

```python
# Training: sample for exploration
output = model.generate(do_sample=True, temperature=0.7)
```

But evaluation typically uses greedy decoding:

```python
# Evaluation: greedy for consistency
output = model.generate(do_sample=False)
```

This creates a mismatch. You might reinforce a sampled output that the model would never produce during greedy evaluation.

### The Solution

Align training with evaluation:

```python
# Option 1: Train on greedy outputs
output = model.generate(do_sample=False)
if reward == 1.0:
    # Reinforce correct greedy outputs
    loss = model(token_ids, labels=token_ids).loss
    loss.backward()

# Option 2: Use PPO (more sophisticated)
# Agent Lightning's VERL provides this
```

---

## Why Strands + strands-vllm Works for Agent Tuning

Strands is a general-purpose agent framework. What makes it useful for training is **strands-vllm** — it exposes vLLM's TITO capability through Strands' callback system.

### 1. The Callback System Captures Training Data

Without Strands, you'd need to manually intercept LLM calls:

```python
# Without Strands: Manual interception (messy)
original_response = llm.generate(prompt)
token_ids = ???  # How do you get these?
```

With Strands, `VLLMTokenRecorder` plugs directly into the event system:

```python
# With Strands: Clean callback integration
from strands import Agent
from strands_vllm import VLLMModel, VLLMTokenRecorder

recorder = VLLMTokenRecorder()
agent = Agent(
    model=VLLMModel(return_token_ids=True),
    callback_handler=recorder,  # ← Automatic capture
)

agent("Solve this problem...")

# Token IDs captured automatically
print(recorder.prompt_token_ids)   # [1, 234, 567, ...]
print(recorder.token_ids)          # [891, 234, 111, ...]
```

### 2. Model Abstraction Lets You Train Locally, Deploy Anywhere

During training, you use vLLM for TITO support. In production, you might use a different backend:

```python
# Training: vLLM for TITO
from strands_vllm import VLLMModel
training_model = VLLMModel(
    base_url="http://localhost:8000/v1",
    return_token_ids=True,
)

# Production: Same agent code, different model
from strands import Agent
from strands.models import BedrockModel
production_model = BedrockModel(model_id="anthropic.claude-3")

# Agent code stays the same!
agent = Agent(model=training_model)  # or production_model
```

### 3. Tool Calling Creates Richer Training Signals

Train agents not just on answers, but on **how they use tools**:

```python
from strands import Agent, tool
from strands_vllm import VLLMModel, VLLMTokenRecorder

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@tool  
def search(query: str) -> str:
    """Search for information."""
    return search_api(query)

recorder = VLLMTokenRecorder()
agent = Agent(
    model=VLLMModel(return_token_ids=True),
    tools=[calculator, search],
    callback_handler=recorder,
)

result = agent("What is the population of France divided by 67?")

# Now you can reward based on:
# - Did the agent use the right tool?
# - Did it call tools in the right order?
# - Was the final answer correct?
```

### 4. Async Support Speeds Up Data Collection

Collecting thousands of rollouts? Parallelize them:

```python
import asyncio
from strands import Agent
from strands_vllm import VLLMModel, VLLMTokenRecorder

async def collect_rollout(problem: dict) -> dict:
    recorder = VLLMTokenRecorder()
    agent = Agent(
        model=VLLMModel(return_token_ids=True),
        callback_handler=recorder,
    )
    result = await agent.invoke_async(problem["question"])
    return {
        "token_ids": recorder.token_ids,
        "reward": compute_reward(result, problem["expected"]),
    }

# Collect 100 rollouts in parallel
problems = load_problems()
rollouts = await asyncio.gather(*[
    collect_rollout(p) for p in problems
])
```

---

## Putting It All Together

Here's the complete flow:

```
1. START vLLM SERVER
   vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16

2. COLLECT DATA (with TITO)
   - Create Strands agent with VLLMModel + VLLMTokenRecorder
   - Run agent on training tasks
   - Capture (prompt_tokens, response_tokens, reward)

3. TRAIN MODEL
   - Load base model with LoRA
   - For each successful rollout:
     - Use exact response_token_ids (no re-tokenization!)
     - Compute loss, update model

4. EVALUATE
   - Test on held-out problems
   - Compare base vs. trained model
```

---

## Key Takeaways

### 1. TITO is Non-Negotiable for RL
Without exact token IDs, you're training on the wrong tokens. Use `strands-vllm` with `return_token_ids=True`.

### 2. Strands Provides the Right Abstractions
The callback system (`VLLMTokenRecorder`) makes TITO capture seamless. Tool calling support lets you train agents for complex tasks.

### 3. Align Training with Evaluation
If you evaluate with greedy decoding, consider training with greedy too. The sampling/greedy mismatch is a common pitfall.

### 4. Start Simple, Scale with Agent Lightning
This guide uses a local setup. For production, Agent Lightning provides the Algorithm/Store/Runner architecture for distributed training.

---

## Try It Yourself

```bash
# Install dependencies
pip install strands-agents strands-vllm torch transformers peft

# Start vLLM
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16

# Run the pipeline
python run_pipeline.py
```

The complete code is available on GitHub: [your repo link]

---

## What's Next?

- **Add tools** — Train agents to use calculators, search, databases
- **Try VERL** — Agent Lightning's PPO implementation for proper on-policy training
- **Scale up** — Use the Algorithm/Store/Runner pattern for distributed rollouts
- **Harder tasks** — GSM8K, coding problems, multi-step reasoning

---

## References

- [Strands Agent SDK](https://github.com/strands-agents/sdk-python)
- [strands-vllm](https://github.com/strands-agents/strands-vllm)
- [Agent Lightning](https://microsoft.github.io/agent-lightning/)
- [Agent Lightning Architecture](https://microsoft.github.io/agent-lightning/stable/deep-dive/birds-eye-view/)
- [TITO Explained](https://microsoft.github.io/agent-lightning/stable/deep-dive/store/)

---

*Building AI agents? Follow for more practical guides on agent development and training.*
