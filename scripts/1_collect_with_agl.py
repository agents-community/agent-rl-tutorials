#!/usr/bin/env python3
"""
Step 1: Data Collection with Agent Lightning + Strands

This uses the ACTUAL Agent Lightning architecture:
- LightningStore: Central database for rollouts and spans
- @agl.rollout: Decorated rollout function
- LitAgentRunner: Executes rollouts and captures spans
- VLLMTokenRecorder: Captures token IDs (TITO)

Prerequisites:
    Terminal 1: agl store              # Start the store
    Terminal 2: vllm serve <model>     # Start vLLM
    Terminal 3: python 1_collect_with_agl.py runner
    Terminal 4: python 1_collect_with_agl.py algo

Reference: https://microsoft.github.io/agent-lightning/stable/deep-dive/birds-eye-view/
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    model_config,
    TRAINING_PROBLEMS,
    SYSTEM_PROMPT,
    format_prompt,
)

import agentlightning as agl
from rich.console import Console

# Strands imports
try:
    from strands import Agent
    from strands_vllm import VLLMModel, VLLMTokenRecorder
except ImportError:
    print("❌ Missing: pip install strands-agents strands-vllm")
    sys.exit(1)

console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

STORE_ENDPOINT = "http://localhost:4747"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", model_config.name)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_strands_model(model_id: str, base_url: str) -> VLLMModel:
    """Create a Strands VLLMModel with TITO support."""
    return VLLMModel(
        base_url=base_url,
        model_id=model_id,
        return_token_ids=True,  # ← TITO: Request token IDs!
        params={"temperature": 0.7},
    )


def extract_answer(text: str) -> Optional[float]:
    """Extract numerical answer from response."""
    match = re.search(r"Answer:\s*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def check_correct(predicted: Optional[float], expected: float) -> bool:
    """Check if prediction matches expected answer."""
    if predicted is None:
        return False
    tolerance = abs(expected) * 0.01 + 0.01
    return abs(predicted - expected) <= tolerance


# =============================================================================
# AGENT LIGHTNING ROLLOUT FUNCTION
# =============================================================================

@agl.rollout
async def math_rollout(
    task: str,
    expected_answer: float,
    llm: agl.LLM,
    rollout: agl.AttemptedRollout,
) -> float:
    """
    Execute a single math problem rollout.
    
    This is the ACTUAL Agent Lightning rollout pattern:
    1. Receives task from store (via Algorithm)
    2. Executes agent with Strands + strands-vllm
    3. Captures token IDs via VLLMTokenRecorder
    4. Emits reward back to store
    
    The @agl.rollout decorator automatically:
    - Tracks the rollout in the store
    - Captures spans from the tracer
    - Associates rewards with the rollout
    """
    # Create Strands model with TITO
    strands_model = create_strands_model(
        model_id=llm.model,
        base_url=llm.endpoint,
    )
    
    # Create token recorder - THIS IS THE TITO MAGIC!
    # It captures exact token IDs from vLLM streaming response
    token_recorder = VLLMTokenRecorder()
    
    # Create Strands Agent
    agent = Agent(
        model=strands_model,
        system_prompt=SYSTEM_PROMPT,
        callback_handler=token_recorder,  # ← Captures token IDs!
    )
    
    try:
        # Execute the agent
        prompt = format_prompt(task)
        result = agent(prompt)
        
        # Extract response text
        if hasattr(result, 'message') and result.message:
            if hasattr(result.message, 'content') and result.message.content:
                response_text = result.message.content[0].text
            else:
                response_text = str(result)
        else:
            response_text = str(result)
        
        # Calculate reward
        predicted = extract_answer(response_text)
        is_correct = check_correct(predicted, expected_answer)
        reward = 1.0 if is_correct else 0.0
        
        # Log TITO status
        has_tito = bool(token_recorder.prompt_token_ids and token_recorder.token_ids)
        tito_status = f"TITO[{len(token_recorder.prompt_token_ids or [])}/{len(token_recorder.token_ids or [])}]"
        
        mark = "✓" if is_correct else "✗"
        console.print(f"  {mark} {task[:40]}... → {predicted} (exp:{expected_answer}) {tito_status}")
        
        # Emit reward to Agent Lightning store
        # This is the KEY connection to the RL training loop!
        agl.reward.emit_reward(reward)
        
        return reward
        
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        agl.reward.emit_reward(0.0)
        return 0.0


# =============================================================================
# AGENT LIGHTNING ALGORITHM
# =============================================================================

@agl.algorithm
async def collect_algorithm(store: agl.LightningStore):
    """
    The Algorithm component of Agent Lightning.
    
    This is the "brain" that:
    1. Enqueues rollouts (tasks) to the store
    2. Waits for runners to complete them
    3. Queries results (spans with token IDs!)
    
    In a full RL setup, this would also update the model.
    """
    console.print("\n" + "="*60)
    console.print("📊 [bold cyan]Agent Lightning Algorithm[/bold cyan]")
    console.print("="*60)
    console.print(f"   Store: {STORE_ENDPOINT}")
    console.print(f"   Model: {VLLM_MODEL}")
    console.print(f"   Problems: {len(TRAINING_PROBLEMS)}")
    
    # Create LLM resource - this tells the runner which model to use
    llm_resource = agl.LLM(
        endpoint=VLLM_BASE_URL,
        model=VLLM_MODEL,
        sampling_parameters={"temperature": 0.7, "max_tokens": 200},
    )
    
    # Update resources in store
    await store.update_resource("llm", llm_resource)
    
    console.print("\n🎯 Enqueuing rollouts...")
    
    # Enqueue all problems as rollouts
    rollout_ids = []
    for i, problem in enumerate(TRAINING_PROBLEMS):
        rollout = await store.enqueue_rollout(
            math_rollout,
            task=problem["q"],
            expected_answer=problem["a"],
            llm=llm_resource,
        )
        rollout_ids.append(rollout.rollout_id)
        console.print(f"   Enqueued [{i+1}/{len(TRAINING_PROBLEMS)}]: {problem['q'][:40]}...")
    
    console.print(f"\n⏳ Waiting for {len(rollout_ids)} rollouts to complete...")
    console.print("   (Make sure runner is running in another terminal)")
    
    # Wait for all rollouts to complete
    completed = await store.wait_for_rollouts(
        rollout_ids=rollout_ids,
        timeout=300,  # 5 minutes
    )
    
    console.print(f"\n✅ Completed: {len(completed)}/{len(rollout_ids)} rollouts")
    
    # Query spans to get token IDs and rewards
    console.print("\n📊 Analyzing results...")
    
    correct_count = 0
    tito_count = 0
    collected_data = []
    
    for rollout in completed:
        # Query spans for this rollout
        spans = await store.query_spans(rollout_ids=[rollout.rollout_id])
        
        # Find reward and token IDs in spans
        reward = agl.reward.find_final_reward(spans)
        
        prompt_token_ids = None
        response_token_ids = None
        
        for span in spans:
            attrs = span.attributes if hasattr(span, 'attributes') else {}
            if "llm.hosted_vllm.prompt_token_ids" in attrs:
                prompt_token_ids = attrs["llm.hosted_vllm.prompt_token_ids"]
            if "llm.hosted_vllm.response_token_ids" in attrs:
                response_token_ids = attrs["llm.hosted_vllm.response_token_ids"]
        
        if reward == 1.0:
            correct_count += 1
        
        has_tito = bool(prompt_token_ids and response_token_ids)
        if has_tito:
            tito_count += 1
        
        collected_data.append({
            "rollout_id": rollout.rollout_id,
            "reward": reward,
            "has_tito": has_tito,
            "prompt_tokens": len(prompt_token_ids) if prompt_token_ids else 0,
            "response_tokens": len(response_token_ids) if response_token_ids else 0,
        })
    
    # Summary
    accuracy = correct_count / len(completed) * 100 if completed else 0
    tito_rate = tito_count / len(completed) * 100 if completed else 0
    
    console.print("\n" + "="*60)
    console.print(f"📈 [bold]Results:[/bold]")
    console.print(f"   Accuracy: {correct_count}/{len(completed)} ({accuracy:.0f}%)")
    console.print(f"   TITO Rate: {tito_count}/{len(completed)} ({tito_rate:.0f}%)")
    console.print("="*60)
    
    # Save collected data
    output_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "collected_data.json"
    )
    
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "model": VLLM_MODEL,
                "total": len(completed),
                "correct": correct_count,
                "accuracy": accuracy,
                "tito_count": tito_count,
                "tito_rate": tito_rate,
                "timestamp": datetime.now().isoformat(),
            },
            "samples": collected_data,
        }, f, indent=2)
    
    console.print(f"\n💾 Saved to: {output_file}")


# =============================================================================
# AGENT LIGHTNING RUNNER
# =============================================================================

@agl.runner
async def run_runner(store: agl.LightningStore):
    """
    The Runner component of Agent Lightning.
    
    This is the "worker" that:
    1. Dequeues rollouts from the store
    2. Executes them using the agent
    3. Sends spans (with token IDs!) back to store
    
    The LitAgentRunner handles all the orchestration.
    """
    console.print("\n" + "="*60)
    console.print("🏃 [bold cyan]Agent Lightning Runner[/bold cyan]")
    console.print("="*60)
    console.print(f"   Store: {STORE_ENDPOINT}")
    console.print(f"   Waiting for rollouts...")
    console.print("   (Press Ctrl+C to stop)")
    
    # Create the runner with AgentOps tracer for span collection
    runner = agl.LitAgentRunner(
        tracer=agl.AgentOpsTracer(),
    )
    
    # Run the agent in a loop, processing rollouts from the store
    with runner.run_context(agent=math_rollout, store=store):
        await runner.iter()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agent Lightning Data Collection with Strands"
    )
    parser.add_argument(
        "role",
        choices=["algo", "runner"],
        help="Role: 'algo' for algorithm, 'runner' for runner"
    )
    args = parser.parse_args()
    
    console.print("\n🚀 [bold]Agent Lightning + Strands + strands-vllm[/bold]")
    
    # Connect to the Agent Lightning store
    store = agl.AgentLightningClient(endpoint=STORE_ENDPOINT)
    
    if args.role == "algo":
        console.print("   Running as: [cyan]Algorithm[/cyan]")
        asyncio.run(collect_algorithm(store))
    else:
        console.print("   Running as: [cyan]Runner[/cyan]")
        asyncio.run(run_runner(store))


if __name__ == "__main__":
    main()

