SHELL   := /bin/bash

# ═══════════════════════════════════════════════════════════
# Config & Variables
# ═══════════════════════════════════════════════════════════

INPUT      ?= data/wikipedia/
OUTPUT     ?= output/baseline
VARIANT    ?= baseline

WIKI_COUNT    ?= 20
WIKI_LANGUAGE ?= en

.PHONY: help install clean test build-kg download-wikipedia

help:
	@echo "Usage: make <target> [INPUT=...] [OUTPUT=...] [VARIANT=baseline]"
	@echo ""
	@echo "── Setup ────────────────────────────────────────────"
	@echo "   install          Sync dependencies with uv + download spaCy model"
	@echo "   clean            Remove generated output folders"
	@echo ""
	@echo "── Pipeline ─────────────────────────────────────────"
	@echo "   build-kg         Run the full pipeline (preprocess → build KG → evaluate → export)"
	@echo ""
	@echo "── Dev ──────────────────────────────────────────────"
	@echo "   test             Run the test suite"
	@echo ""
	@echo "── Quick Start ──────────────────────────────────────"
	@echo "   make install                                  # one-time setup"
	@echo "   make test                                     # verify everything works"
	@echo "   make build-kg                                 # baseline pipeline"
	@echo "   make build-kg INPUT=data/my_corpus/ OUTPUT=output/exp1/"

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
