# Complete RL Pipeline with Agent Lightning + Strands

This demonstrates the **official Agent Lightning architecture** with Strands Agent SDK and strands-vllm for TITO (Token-In-Token-Out).

## 🏗️ Architecture

Based on [Agent Lightning's Bird's Eye View](https://microsoft.github.io/agent-lightning/stable/deep-dive/birds-eye-view/):

```
┌──────────────────────────────────────────────────────────────────┐
│                    AGENT LIGHTNING ARCHITECTURE                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐           ┌─────────────┐         ┌──────────┐ │
│  │  Algorithm  │──enqueue──│   Store     │──dequeue│  Runner  │ │
│  │  (brain)    │           │  (database) │         │ (worker) │ │
│  └─────────────┘           └─────────────┘         └──────────┘ │
│        │                         │                       │       │
│        │ update_resources        │ spans               │ execute │
│        └─────────────────────────┘                     │         │
│                                                        ▼         │
│                                                  ┌──────────┐   │
│                                                  │  Strands │   │
│                                                  │  Agent   │   │
│                                                  └──────────┘   │
│                                                        │         │
│                                                        ▼         │
│                                                  ┌──────────┐   │
│                                                  │strands-  │   │
│                                                  │  vllm    │   │
│                                                  │ (TITO!)  │   │
│                                                  └──────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
complete_pipeline/
├── README.md                      # This file
├── run_pipeline.py                # Simplified runner (no vLLM needed)
├── configs/
│   └── config.py                  # All configuration
├── scripts/
│   ├── 1_collect_with_agl.py      # ✨ FULL Agent Lightning + Strands
│   ├── 1_collect_data.py          # Standalone Strands demo
│   ├── 2_train_rl.py              # REINFORCE training
│   └── 3_evaluate.py              # Model comparison
└── docs/
    └── ARTICLE_NOTES.md           # Notes for your article
```

## 🚀 Option 1: Full Agent Lightning Pipeline (Recommended)

This uses the **actual Agent Lightning components**.

### Prerequisites

```bash
# Install dependencies
pip install agentlightning strands-agents strands-vllm

# Start vLLM (in a separate terminal)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --port 8000
```

### Run with 4 Terminals

```bash
# Terminal 1: Start Agent Lightning Store
agl store

# Terminal 2: Start vLLM (if not already running)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16

# Terminal 3: Start the Runner (worker)
cd complete_pipeline
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
python scripts/1_collect_with_agl.py runner

# Terminal 4: Run the Algorithm (brain)
cd complete_pipeline
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
python scripts/1_collect_with_agl.py algo
```

### What Happens

1. **Algorithm** enqueues math problems to the Store
2. **Runner** dequeues and executes them with Strands Agent
3. **VLLMTokenRecorder** captures token IDs (TITO!)
4. **Spans** (with token IDs) flow back to Store
5. **Algorithm** queries results and calculates accuracy

## 🚀 Option 2: Simplified Pipeline (No vLLM)

For quick demos without external dependencies:

```bash
cd complete_pipeline

# Run training and evaluation
python run_pipeline.py --skip-collect
```

## 🔑 Key Concepts

### TITO (Token-In-Token-Out)

```python
from strands_vllm import VLLMModel, VLLMTokenRecorder

# Request token IDs from vLLM
model = VLLMModel(return_token_ids=True)

# Capture them during streaming
recorder = VLLMTokenRecorder()
agent = Agent(model=model, callback_handler=recorder)

# After agent runs:
print(recorder.prompt_token_ids)   # Exact input tokens
print(recorder.token_ids)          # Exact output tokens
```

### Agent Lightning Decorators

```python
import agentlightning as agl

@agl.rollout
async def my_rollout(task: str, llm: agl.LLM) -> float:
    # Execute agent
    result = agent(task)
    
    # Emit reward to store
    agl.reward.emit_reward(1.0 if correct else 0.0)
    return reward

@agl.algorithm
async def my_algorithm(store: agl.LightningStore):
    # Enqueue rollouts
    await store.enqueue_rollout(my_rollout, task="...")
    
    # Wait and query results
    completed = await store.wait_for_rollouts(rollout_ids)
    spans = await store.query_spans(rollout_ids)
```

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| Accuracy | ~80% |
| TITO Rate | 100% (with vLLM) |
| Token IDs | Captured in spans |

## 🔗 References

- [Agent Lightning Docs](https://microsoft.github.io/agent-lightning/)
- [Bird's Eye View Architecture](https://microsoft.github.io/agent-lightning/stable/deep-dive/birds-eye-view/)
- [Strands Agent SDK](https://github.com/strands-agents/sdk-python)
- [strands-vllm](https://github.com/strands-agents/strands-vllm)
