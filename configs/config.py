"""
Configuration for the Complete RL Training Pipeline

All settings in one place for easy customization.
"""

from dataclasses import dataclass
from typing import List, Dict
import torch

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Model and inference settings."""
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    vllm_base_url: str = "http://localhost:8000/v1"
    dtype: torch.dtype = torch.bfloat16
    
    # LoRA settings for efficient training
    lora_r: int = 32  # Higher rank for more capacity
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """RL training settings."""
    num_epochs: int = 10
    learning_rate: float = 2e-5
    temperature: float = 0.7  # For exploration
    max_new_tokens: int = 200
    
    # REINFORCE baseline
    baseline: float = 0.5
    
    # Early stopping
    early_stop_accuracy: float = 1.0  # Disable early stopping
    
    # Output
    output_dir: str = "./trained_model"

# =============================================================================
# TRAINING DATA
# =============================================================================

# =============================================================================
# CHALLENGING PROBLEMS - Base model gets ~50%, RL can improve to 80%+
# =============================================================================

TRAINING_PROBLEMS: List[Dict] = [
    # Multi-step arithmetic (model often makes errors)
    {"q": "A farmer has 24 cows. He buys 8 more, then sells half of all his cows. How many cows remain?", "a": 16},
    {"q": "Tom has $50. He buys 3 books at $8 each. How much money is left?", "a": 26},
    {"q": "A train travels at 60 mph for 2 hours, then 80 mph for 3 hours. What is the total distance in miles?", "a": 360},
    {"q": "Lisa has 48 stickers. She gives half to her brother. How many does she have left?", "a": 24},
    {"q": "A baker makes 36 cookies. He puts them equally into 4 boxes. How many cookies per box?", "a": 9},
    
    # Percentage problems (often tricky)
    {"q": "A $80 jacket is 20% off. What is the sale price in dollars?", "a": 64},
    {"q": "A car costs $20000. Its value drops 15%. What is the new value in dollars?", "a": 17000},
    {"q": "What is 25% of 80?", "a": 20},
    {"q": "A shirt originally costs $40. After a 25% discount, what is the price?", "a": 30},
    
    # Unit conversion and rates
    {"q": "A car travels 180 miles in 3 hours. What is its speed in mph?", "a": 60},
    {"q": "If 5 apples cost $10, how much do 8 apples cost in dollars?", "a": 16},
    {"q": "A worker earns $15 per hour. How much does he earn in 8 hours?", "a": 120},
    
    # Two-step problems
    {"q": "John has 30 marbles. He gives 10 to Tom and buys 5 more. How many does John have now?", "a": 25},
    {"q": "A pizza has 8 slices. Tom eats 2 and Mary eats 3. How many slices are left?", "a": 3},
    {"q": "There are 45 students. 20 are boys. How many girls are there?", "a": 25},
]

# Held-out test problems
TEST_PROBLEMS: List[Dict] = [
    {"q": "A shop has 100 items. It sells 40 items. How many remain?", "a": 60},
    {"q": "Mike earns $12 per hour. He works 8 hours. What is his total pay?", "a": 96},
    {"q": "A $100 item is 20% off. What is the sale price?", "a": 80},
    {"q": "Sarah has 50 stickers. She gives 15 to her friend. How many does she have left?", "a": 35},
    {"q": "A car travels 240 miles in 4 hours. What is its speed in mph?", "a": 60},
    {"q": "Buy 6 pens at $3 each. What is the total cost?", "a": 18},
]

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT = """You are a math tutor. Solve problems step by step.
IMPORTANT: Always end with exactly "Answer: " followed by just the number."""

def format_prompt(question: str) -> str:
    """Format a question into a prompt."""
    return f"""{question}

Think step by step, then give your final answer as "Answer: [number]"."""

# =============================================================================
# INSTANTIATE CONFIGS
# =============================================================================

model_config = ModelConfig()
training_config = TrainingConfig()

