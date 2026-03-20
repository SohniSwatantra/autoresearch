#!/bin/bash
set -e

echo "=============================================="
echo "MCP Autoresearch Pipeline"
echo "Started: $(date)"
echo "=============================================="

# Phase 1: Crawl MCP ecosystem
echo ""
echo "=== PHASE 1: Crawling MCP ecosystem ==="
echo "This collects data from GitHub, npm, PyPI, Reddit, etc."
echo ""
python3 mcp_researcher.py

echo ""
echo "Phase 1 complete. Results in results/ and corpus/"
echo ""

# Phase 2: Prepare training data from crawled corpus
echo "=== PHASE 2: Preparing training data ==="
echo "Training BPE tokenizer and creating parquet shards..."
echo ""
python3 prepare_mcp.py

echo ""
echo "Phase 2 complete. Training data ready."
echo ""

# Phase 3: Patch train.py to use MCP data
echo "=== PHASE 3: Patching train.py for MCP training ==="
echo ""

# Replace the import line to use prepare_mcp instead of prepare
if grep -q "from prepare import" train.py; then
    sed -i 's/from prepare import/from prepare_mcp import/' train.py
    echo "Patched train.py to import from prepare_mcp"
else
    echo "train.py already patched or uses different import"
fi

echo ""

# Phase 4: Autonomous training loop
echo "=== PHASE 4: Starting autoresearch training loop ==="
echo "Creating experiment branch..."
echo ""

BRANCH="autoresearch/mcp-$(date +%Y%m%d)"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
echo "On branch: $BRANCH"
echo ""

echo "Starting autonomous agent (Qwen via ollama)..."
echo "Press Ctrl+C to stop."
echo ""
python3 agent.py
