"""
Autonomous Training Agent - Phase 2 of the autoresearch pipeline.
Replaces Claude Code in the autoresearch loop, using Qwen via ollama.

Reads program.md for instructions, proposes modifications to train.py,
validates, runs experiments, and keeps/discards based on val_bpb.

Usage: python3 agent.py
"""

import os
import re
import ast
import sys
import time
import signal
import subprocess
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
TRAIN_SCRIPT = "train.py"
RESULTS_FILE = "results.tsv"
RUN_LOG = "run.log"
RUN_TIMEOUT = 600  # 10 minutes max per experiment
MAX_CONSECUTIVE_CRASHES = 3

# ---------------------------------------------------------------------------
# Ollama interface
# ---------------------------------------------------------------------------

def query_qwen(prompt, max_tokens=4096):
    """Query Qwen via ollama. Returns the response text."""
    import requests
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
            }
        }, timeout=180)
        if resp.ok:
            return resp.json().get("response", "")
        else:
            print(f"Ollama error: {resp.status_code} {resp.text[:200]}")
            return ""
    except Exception as e:
        print(f"Ollama connection error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_run(*args):
    """Run a git command, return stdout."""
    result = subprocess.run(["git"] + list(args), capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def git_commit(message):
    """Stage train.py and commit."""
    subprocess.run(["git", "add", TRAIN_SCRIPT], check=True, timeout=30)
    subprocess.run(["git", "commit", "-m", message], check=True, timeout=30)
    return git_run("rev-parse", "--short", "HEAD")


def git_reset_hard(commit):
    """Reset to a specific commit."""
    subprocess.run(["git", "reset", "--hard", commit], check=True, timeout=30)


def get_current_commit():
    """Get current short commit hash."""
    return git_run("rev-parse", "--short", "HEAD")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment():
    """Run train.py and return (val_bpb, peak_vram_mb) or (None, None) on failure."""
    print(f"  Running experiment... (timeout: {RUN_TIMEOUT}s)")
    try:
        with open(RUN_LOG, "w") as log_file:
            proc = subprocess.run(
                ["uv", "run", TRAIN_SCRIPT],
                stdout=log_file, stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT,
            )
        if proc.returncode != 0:
            print(f"  Experiment crashed (exit code {proc.returncode})")
            return None, None
    except subprocess.TimeoutExpired:
        print(f"  Experiment timed out after {RUN_TIMEOUT}s")
        return None, None
    except Exception as e:
        print(f"  Experiment error: {e}")
        return None, None

    # Parse results
    val_bpb = None
    peak_vram = None
    try:
        with open(RUN_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("val_bpb:"):
                    val_bpb = float(line.split(":")[1].strip())
                elif line.startswith("peak_vram_mb:"):
                    peak_vram = float(line.split(":")[1].strip())
    except Exception as e:
        print(f"  Error parsing run.log: {e}")

    return val_bpb, peak_vram


def get_crash_info():
    """Get last 50 lines of run.log for crash diagnosis."""
    try:
        with open(RUN_LOG, "r") as f:
            lines = f.readlines()
        return "".join(lines[-50:])
    except Exception:
        return "Could not read run.log"


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def init_results():
    """Initialize results.tsv if it doesn't exist."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")


def log_result(commit, val_bpb, memory_gb, status, description):
    """Append a result to results.tsv."""
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  Logged: {commit} | val_bpb={val_bpb:.6f} | {memory_gb:.1f}GB | {status} | {description}")


def get_results_history():
    """Read results.tsv and return as string."""
    if not os.path.exists(RESULTS_FILE):
        return "No results yet."
    with open(RESULTS_FILE, "r") as f:
        return f.read()


def get_best_bpb():
    """Get the best (lowest) val_bpb from results history."""
    best = float("inf")
    if not os.path.exists(RESULTS_FILE):
        return best
    with open(RESULTS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[3] == "keep":
                try:
                    bpb = float(parts[1])
                    if bpb > 0 and bpb < best:
                        best = bpb
                except ValueError:
                    continue
    return best


# ---------------------------------------------------------------------------
# Code modification
# ---------------------------------------------------------------------------

def read_train_py():
    """Read current train.py."""
    with open(TRAIN_SCRIPT, "r") as f:
        return f.read()


def write_train_py(code):
    """Write modified train.py."""
    with open(TRAIN_SCRIPT, "w") as f:
        f.write(code)


def validate_syntax(code):
    """Check if Python code is syntactically valid."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def extract_code_from_response(response):
    """Extract Python code block from Qwen's response."""
    # Look for code blocks
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        # Return the longest code block (likely the full file)
        return max(blocks, key=len)
    # If no code blocks, try to find the full file content
    if "import " in response and "def " in response:
        return response
    return None


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

def build_experiment_prompt(train_code, results_history, best_bpb, crash_info=None):
    """Build the prompt for Qwen to propose an experiment."""
    prompt = f"""You are an autonomous ML researcher optimizing a GPT training script.

GOAL: Lower val_bpb (bits per byte on validation set). Current best: {best_bpb:.6f}

CONSTRAINTS:
- Only modify train.py (the file below)
- Cannot modify prepare.py or prepare_mcp.py (data loading, evaluation are fixed)
- Cannot install new packages
- Training runs for a fixed 5-minute time budget
- GPU has 48GB VRAM (shared with this Qwen model using ~12GB)
- Available VRAM for training: ~35GB

CURRENT train.py:
```python
{train_code}
```

EXPERIMENT HISTORY:
{results_history}

{"LAST CRASH:" + chr(10) + crash_info if crash_info else ""}

INSTRUCTIONS:
1. Analyze the code and history
2. Propose ONE specific modification to improve val_bpb
3. Explain your reasoning briefly
4. Output the COMPLETE modified train.py in a Python code block

Focus on:
- Model architecture (depth, width, attention patterns)
- Optimizer hyperparameters (learning rates, betas, weight decay)
- Batch size and gradient accumulation
- Activation functions
- Any other architectural innovation

Be bold but practical. Small, targeted changes are often better than large rewrites.

Output the complete modified train.py:"""
    return prompt


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Autonomous Training Agent (Qwen via ollama)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    init_results()
    consecutive_crashes = 0
    experiment_num = 0

    # Check if baseline exists
    results = get_results_history()
    if "baseline" not in results.lower():
        print("\n--- Experiment 0: Baseline ---")
        base_commit = get_current_commit()
        val_bpb, peak_vram = run_experiment()
        if val_bpb is not None:
            memory_gb = peak_vram / 1024 if peak_vram else 0
            log_result(base_commit, val_bpb, memory_gb, "keep", "baseline")
            print(f"  Baseline val_bpb: {val_bpb:.6f}")
        else:
            crash_info = get_crash_info()
            print(f"  Baseline run failed! Check run.log")
            print(f"  Last lines: {crash_info[-500:]}")
            log_result(base_commit, 0.0, 0.0, "crash", "baseline failed")
            print("  Continuing anyway to let Qwen try fixes...")
        experiment_num = 1

    # Main loop
    while True:
        print(f"\n{'=' * 60}")
        print(f"--- Experiment {experiment_num} ---")
        print(f"Time: {datetime.now().isoformat()}")

        best_bpb = get_best_bpb()
        base_commit = get_current_commit()
        train_code = read_train_py()
        results_history = get_results_history()

        # Build crash context if last run crashed
        crash_context = None
        if consecutive_crashes > 0:
            crash_context = get_crash_info()

        # Ask Qwen for a modification
        print("  Querying Qwen for experiment proposal...")
        prompt = build_experiment_prompt(train_code, results_history, best_bpb, crash_context)
        response = query_qwen(prompt, max_tokens=8192)

        if not response:
            print("  Qwen returned empty response, waiting 30s...")
            time.sleep(30)
            experiment_num += 1
            continue

        # Extract and validate code
        new_code = extract_code_from_response(response)
        if not new_code:
            print("  Could not extract code from Qwen's response")
            # Log what Qwen said for debugging
            print(f"  Response preview: {response[:300]}...")
            experiment_num += 1
            continue

        valid, error = validate_syntax(new_code)
        if not valid:
            print(f"  Syntax error in proposed code: {error}")
            experiment_num += 1
            consecutive_crashes += 1
            if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                print(f"  {MAX_CONSECUTIVE_CRASHES} consecutive failures, resetting to base...")
                git_reset_hard(base_commit)
                consecutive_crashes = 0
            continue

        # Apply modification
        # Extract description from Qwen's response (first non-code line)
        desc_lines = [l.strip() for l in response.split("\n") if l.strip() and not l.strip().startswith("```") and not l.strip().startswith("import")]
        description = desc_lines[0][:100] if desc_lines else f"experiment {experiment_num}"

        write_train_py(new_code)
        try:
            commit_hash = git_commit(f"exp{experiment_num}: {description}")
        except Exception as e:
            print(f"  Git commit failed: {e}")
            git_reset_hard(base_commit)
            experiment_num += 1
            continue

        # Run experiment
        val_bpb, peak_vram = run_experiment()

        if val_bpb is None:
            # Crash
            crash_info = get_crash_info()
            print(f"  CRASH: {crash_info[-200:]}")
            log_result(commit_hash, 0.0, 0.0, "crash", description)
            git_reset_hard(base_commit)
            consecutive_crashes += 1

            if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                print(f"  {MAX_CONSECUTIVE_CRASHES} consecutive crashes, resetting state...")
                consecutive_crashes = 0
        else:
            consecutive_crashes = 0
            memory_gb = peak_vram / 1024 if peak_vram else 0

            if val_bpb < best_bpb:
                # Improvement! Keep it.
                log_result(commit_hash, val_bpb, memory_gb, "keep", description)
                print(f"  KEEP: {val_bpb:.6f} < {best_bpb:.6f} (improved by {best_bpb - val_bpb:.6f})")
            else:
                # No improvement, discard.
                log_result(commit_hash, val_bpb, memory_gb, "discard", description)
                print(f"  DISCARD: {val_bpb:.6f} >= {best_bpb:.6f}")
                git_reset_hard(base_commit)

        experiment_num += 1
        print(f"  Best val_bpb so far: {get_best_bpb():.6f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAgent stopped by user.")
        print(f"Final results:\n{get_results_history()}")
        sys.exit(0)
