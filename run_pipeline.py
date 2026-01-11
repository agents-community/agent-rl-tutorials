#!/usr/bin/env python3
"""
Complete RL Training Pipeline - One Command to Run Everything

Usage:
    python run_pipeline.py              # Run full pipeline
    python run_pipeline.py --skip-collect  # Skip data collection (use existing)
    python run_pipeline.py --only train    # Only run training
    python run_pipeline.py --only eval     # Only run evaluation
"""

import argparse
import subprocess
import sys
import os

from rich.console import Console
from rich.panel import Panel

console = Console()

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")


def print_banner():
    """Print the pipeline banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🚀 Complete RL Training Pipeline                               ║
║                                                                  ║
║   Strands Agent SDK + strands-vllm + RL Training                 ║
║                                                                  ║
║   Step 1: Collect data with TITO (Token-In-Token-Out)           ║
║   Step 2: Train with REINFORCE (Policy Gradients)               ║
║   Step 3: Evaluate and compare Base vs Trained                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")


def run_step(script_name: str, step_name: str) -> bool:
    """Run a pipeline step."""
    console.print(f"\n{'='*60}")
    console.print(f"🔄 Running: [bold]{step_name}[/bold]")
    console.print(f"   Script: {script_name}")
    console.print('='*60)
    
    script_path = os.path.join(SCRIPT_DIR, script_name)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"\n❌ Step failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        console.print(f"\n❌ Script not found: {script_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the complete RL training pipeline")
    parser.add_argument(
        "--skip-collect", 
        action="store_true",
        help="Skip data collection (use existing data)"
    )
    parser.add_argument(
        "--only",
        choices=["collect", "train", "eval"],
        help="Only run a specific step"
    )
    args = parser.parse_args()
    
    print_banner()
    
    # Check for vLLM (only for collect step)
    if args.only == "collect" or (not args.skip_collect and args.only is None):
        console.print("\n⚠️  Make sure vLLM is running for data collection:")
        console.print("   [dim]vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16[/dim]")
        console.print("")
    
    steps = []
    
    if args.only:
        if args.only == "collect":
            steps = [("1_collect_data.py", "Step 1: Data Collection with TITO")]
        elif args.only == "train":
            steps = [("2_train_rl.py", "Step 2: RL Training with REINFORCE")]
        elif args.only == "eval":
            steps = [("3_evaluate.py", "Step 3: Model Evaluation")]
    else:
        if not args.skip_collect:
            steps.append(("1_collect_data.py", "Step 1: Data Collection with TITO"))
        steps.append(("2_train_rl.py", "Step 2: RL Training with REINFORCE"))
        steps.append(("3_evaluate.py", "Step 3: Model Evaluation"))
    
    # Run steps
    for script, name in steps:
        success = run_step(script, name)
        if not success:
            console.print(f"\n❌ Pipeline stopped at: {name}")
            sys.exit(1)
    
    # Final summary
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold green]✅ Pipeline Complete![/bold green]\n\n"
        "Results saved to:\n"
        "  • collected_data.json (TITO data)\n"
        "  • trained_model/ (LoRA adapter)\n"
        "  • evaluation_results.json (comparison)",
        title="Success",
        border_style="green"
    ))
    
    console.print("\n📝 For your article, see: [cyan]docs/ARTICLE_NOTES.md[/cyan]")


if __name__ == "__main__":
    main()

