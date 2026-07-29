SHELL   := /bin/bash

# ═══════════════════════════════════════════════════════════
# Config & Variables
# ═══════════════════════════════════════════════════════════

INPUT      ?= data/wikipedia/
OUTPUT     ?= output/baseline
VARIANT    ?= baseline
EXP        ?= experiments/kg/001_baseline/config.yaml

WIKI_COUNT    ?= 20
WIKI_LANGUAGE ?= en

.PHONY: help install clean test build-kg run-experiment download-wikipedia

help:
	@echo "Usage: make <target> [INPUT=...] [OUTPUT=...] [VARIANT=baseline] [EXP=...] [WIKI_COUNT=20]"
	@echo ""
	@echo "── Setup ────────────────────────────────────────────"
	@echo "   install           Sync dependencies with uv + download spaCy model"
	@echo "   clean             Remove generated output folders"
	@echo ""
	@echo "── Data ─────────────────────────────────────────────"
	@echo "   download-wikipedia  Download random Wikipedia articles as JSONL"
	@echo ""
	@echo "── Pipeline ─────────────────────────────────────────"
	@echo "   build-kg          Run the full pipeline (preprocess → build KG → evaluate → export)"
	@echo "   run-experiment    Run an experiment from a YAML config (set EXP= path)"
	@echo ""
	@echo "── Dev ──────────────────────────────────────────────"
	@echo "   test              Run the test suite"
	@echo ""
	@echo "── Quick Start ──────────────────────────────────────"
	@echo "   make install                                   # one-time setup"
	@echo "   make test                                      # verify everything works"
	@echo "   make build-kg                                  # baseline pipeline (direct)"
	@echo "   make run-experiment                            # baseline experiment (001)"
	@echo "   make run-experiment EXP=experiments/kg/002_llm/config.yaml"
	@echo "   make download-wikipedia WIKI_COUNT=50          # download 50 articles"

# ═══════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════

## install: Set up the project and install dependencies
install:
	uv sync
	uv run python -m spacy download en_core_web_sm

## clean: Remove generated output folders
clean:
	rm -rf output/ generated_KGs/

# ═══════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════

## build-kg: Run the full pipeline (preprocess → build KG → evaluate → export)
build-kg:
	uv run python main.py -i $(INPUT) -o $(OUTPUT) --variant $(VARIANT)

## run-experiment: Run an experiment from a YAML config file
run-experiment:
	uv run python main.py --experiment $(EXP)

# ═══════════════════════════════════════════════════════════
# Dev
# ═══════════════════════════════════════════════════════════

## test: Run the test suite
test:
	uv run pytest tests/ -v

## download-wikipedia: Download random Wikipedia articles as Polygraph JSONL
download-wikipedia:
	uv run python tools/data_retrieval/download_wikipedia.py \
		--count $(WIKI_COUNT) \
		--language $(WIKI_LANGUAGE)
