#!/usr/bin/env python3
"""
Step 3: Evaluation - Compare Base vs Trained Model

This script demonstrates the improvement from RL training:
1. Load the BASE model (before training)
2. Load the TRAINED model (after RL training)
3. Compare performance on held-out test problems

Key Insight:
    The trained model should perform better because:
    - It learned from its own attempts (not examples)
    - It was reinforced for correct answers
    - It was discouraged from wrong answers
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# Add parent to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    model_config,
    training_config,
    TEST_PROBLEMS,
    TRAINING_PROBLEMS,
    format_prompt,
)

import torch
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_answer(text: str) -> Optional[float]:
    """Extract numerical answer from response text."""
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
# EVALUATION
# =============================================================================

def evaluate_model(
    model: Any, 
    tokenizer: Any, 
    problems: List[Dict], 
    model_name: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """Evaluate a model on a set of problems."""
    
    if verbose:
        console.print(f"\n📊 Evaluating: [bold]{model_name}[/bold]")
        console.print("-"*50)
    
    model.eval()
    correct = 0
    results = []
    
    for i, problem in enumerate(problems):
        question = problem["q"]
        expected = problem["a"]
        prompt = format_prompt(question)
        
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=False,  # Greedy for consistent evaluation
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        predicted = extract_answer(response)
        is_correct = check_correct(predicted, expected)
        
        if is_correct:
            correct += 1
            mark = "✓"
        else:
            mark = "✗"
        
        if verbose:
            console.print(f"  {mark} {question[:35]:35}... → {predicted} (exp:{expected})")
        
        results.append({
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:100],
        })
    
    accuracy = correct / len(problems) * 100
    
    if verbose:
        console.print(f"\n   Accuracy: {correct}/{len(problems)} ({accuracy:.0f}%)")
    
    return {
        "model": model_name,
        "correct": correct,
        "total": len(problems),
        "accuracy": accuracy,
        "results": results,
    }


def compare_models():
    """Compare base model vs trained model."""
    
    console.print("\n" + "="*60)
    console.print("📊 [bold cyan]Step 3: Model Comparison[/bold cyan]")
    console.print("="*60)
    
    # =========================================================================
    # LOAD TOKENIZER
    # =========================================================================
    
    console.print(f"\n📦 Loading tokenizer: [green]{model_config.name}[/green]")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # =========================================================================
    # EVALUATE BASE MODEL
    # =========================================================================
    
    console.print("\n📦 Loading [yellow]BASE[/yellow] model (before training)...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        device_map="auto",
        torch_dtype=model_config.dtype,
        trust_remote_code=True,
    )
    
    # Evaluate on training problems
    console.print("\n[bold]On Training Problems:[/bold]")
    base_train_results = evaluate_model(
        base_model, tokenizer, TRAINING_PROBLEMS, 
        "Base Model (Training Set)"
    )
    
    # Evaluate on test problems
    console.print("\n[bold]On Test Problems (held-out):[/bold]")
    base_test_results = evaluate_model(
        base_model, tokenizer, TEST_PROBLEMS,
        "Base Model (Test Set)"
    )
    
    # Free memory
    del base_model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # EVALUATE TRAINED MODEL
    # =========================================================================
    
    trained_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        training_config.output_dir
    )
    
    if not os.path.exists(trained_dir):
        console.print(f"\n❌ Trained model not found at: {trained_dir}")
        console.print("   Run [cyan]python scripts/2_train_rl.py[/cyan] first.")
        return None
    
    console.print(f"\n📦 Loading [green]TRAINED[/green] model from: {trained_dir}")
    
    # Load base model again
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        device_map="auto",
        torch_dtype=model_config.dtype,
        trust_remote_code=True,
    )
    
    # Apply LoRA adapter
    trained_model = PeftModel.from_pretrained(base_model, trained_dir)
    
    # Evaluate on training problems
    console.print("\n[bold]On Training Problems:[/bold]")
    trained_train_results = evaluate_model(
        trained_model, tokenizer, TRAINING_PROBLEMS,
        "Trained Model (Training Set)"
    )
    
    # Evaluate on test problems
    console.print("\n[bold]On Test Problems (held-out):[/bold]")
    trained_test_results = evaluate_model(
        trained_model, tokenizer, TEST_PROBLEMS,
        "Trained Model (Test Set)"
    )
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    
    console.print("\n" + "="*60)
    console.print("[bold]📊 COMPARISON RESULTS[/bold]")
    console.print("="*60)
    
    table = Table(title="Model Performance Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Training Set", justify="center")
    table.add_column("Test Set", justify="center")
    
    table.add_row(
        "Base (before RL)",
        f"{base_train_results['correct']}/{base_train_results['total']} ({base_train_results['accuracy']:.0f}%)",
        f"{base_test_results['correct']}/{base_test_results['total']} ({base_test_results['accuracy']:.0f}%)",
    )
    
    table.add_row(
        "Trained (after RL)",
        f"[green]{trained_train_results['correct']}/{trained_train_results['total']} ({trained_train_results['accuracy']:.0f}%)[/green]",
        f"[green]{trained_test_results['correct']}/{trained_test_results['total']} ({trained_test_results['accuracy']:.0f}%)[/green]",
    )
    
    console.print(table)
    
    # Calculate improvements
    train_improvement = trained_train_results['accuracy'] - base_train_results['accuracy']
    test_improvement = trained_test_results['accuracy'] - base_test_results['accuracy']
    
    console.print("\n[bold]Improvements:[/bold]")
    
    if train_improvement > 0:
        console.print(f"   Training Set: [green]+{train_improvement:.0f}%[/green] 🎉")
    elif train_improvement < 0:
        console.print(f"   Training Set: [red]{train_improvement:.0f}%[/red] ⚠️")
    else:
        console.print(f"   Training Set: No change")
    
    if test_improvement > 0:
        console.print(f"   Test Set:     [green]+{test_improvement:.0f}%[/green] 🎉")
    elif test_improvement < 0:
        console.print(f"   Test Set:     [red]{test_improvement:.0f}%[/red] ⚠️")
    else:
        console.print(f"   Test Set:     No change")
    
    console.print("="*60)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    results_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluation_results.json"
    )
    
    full_results = {
        "base_model": {
            "training_set": base_train_results,
            "test_set": base_test_results,
        },
        "trained_model": {
            "training_set": trained_train_results,
            "test_set": trained_test_results,
        },
        "improvements": {
            "training_set": train_improvement,
            "test_set": test_improvement,
        },
    }
    
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    
    console.print(f"\n💾 Results saved to: [green]{results_file}[/green]")
    
    return full_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    console.print("\n🚀 [bold]Model Comparison[/bold]")
    console.print("   Base vs RL-Trained")
    
    try:
        results = compare_models()
        if results:
            console.print("\n✅ [bold green]Step 3 Complete![/bold green]")
            console.print("\n🎉 [bold]Pipeline finished successfully![/bold]")
    except Exception as e:
        console.print(f"\n❌ Error: {e}")
        raise

