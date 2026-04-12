#!/bin/bash
# Deploy Entropic Atlas to HuggingFace Spaces
#
# Usage: ./deploy_to_hf.sh
#
# Prerequisites:
#   1. Install HF CLI: pip install huggingface_hub
#   2. Login: huggingface-cli login  (use a write token from https://huggingface.co/settings/tokens)

set -e

SPACE_REPO="https://huggingface.co/spaces/Arun0808/entropic-atlas"
WORK_DIR=$(mktemp -d)

echo "=== Deploying Entropic Atlas to HuggingFace Spaces ==="
echo "Cloning Space repo..."
git clone "$SPACE_REPO" "$WORK_DIR/space"

echo "Copying project files..."
# Copy project files, excluding local-only and build artifacts.
# The Dockerfile rebuilds from pyproject.toml so .venv/ is never needed.
rsync -av \
  --exclude='.git' \
  --exclude='.github' \
  --exclude='.venv' \
  --exclude='.claude' \
  --exclude='.pytest_cache' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='tests' \
  --exclude='scenarios' \
  "$(dirname "$0")/" "$WORK_DIR/space/"

# Create the Space-specific README (overwrites the project README)
cat > "$WORK_DIR/space/README.md" << 'SPACE_README'
---
title: Entropic Atlas
sdk: docker
app_port: 9019
colorFrom: purple
colorTo: indigo
pinned: false
---

# Entropic Atlas

Spatial-aware research agent for AgentX-AgentBeats Phase 2 Sprint 2 — Research Agent Track.

Handles **FieldWorkArena** (multimodal spatial QA) and **MLE-Bench** (75 Kaggle competitions) via A2A protocol.

Source: [github.com/arunshar/entropic-atlas](https://github.com/arunshar/entropic-atlas)
SPACE_README

echo "Pushing to HuggingFace Space..."
cd "$WORK_DIR/space"
git add -A
git commit -m "Deploy entropic-atlas agent"
git push

echo ""
echo "=== Done! ==="
echo "Space URL: https://huggingface.co/spaces/Arun0808/entropic-atlas"
echo "Agent Card: https://Arun0808-entropic-atlas.hf.space/.well-known/agent-card.json"
echo ""
echo "Don't forget to add your OPENAI_API_KEY as a Space Secret:"
echo "  Settings > Repository secrets > New secret > OPENAI_API_KEY"

# Cleanup
rm -rf "$WORK_DIR"
