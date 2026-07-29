SHELL   := /bin/bash

# ═══════════════════════════════════════════════════════════
# Config & Variables
# ═══════════════════════════════════════════════════════════

INPUT      ?= data/wikipedia/
OUTPUT     ?= output/baseline
VARIANT    ?= baseline
EXP        ?= kg/001_baseline

WIKI_COUNT    ?= 3
WIKI_LANGUAGE ?= en
WIKI_SNAPSHOT ?= 20231101
WIKI_OUTPUT   ?= data/wikipedia/random_articles.jsonl
WIKI_MAX_SCAN ?= 10000
WIKI_SEED     ?=

GRAPH         ?= kg/001_baseline

.PHONY: help install clean test build-kg experiment neo4j-upload download-wikipedia

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
	@echo "   experiment        Run an experiment from a YAML config (set EXP= path)"
	@echo "   neo4j-upload      Upload a knowledge_graph.json to Neo4j (clears first)"
	@echo ""
	@echo "── Dev ──────────────────────────────────────────────"
	@echo "   test              Run the test suite"
	@echo ""
	@echo "── Quick Start ──────────────────────────────────────"
	@echo "   make install                                   # one-time setup"
	@echo "   make test                                      # verify everything works"
	@echo "   make build-kg                                  # baseline pipeline (direct)"
	@echo "   make experiment                                # baseline experiment (001)"
	@echo "   make experiment EXP=kg/002_llm"
	@echo "   make neo4j-upload GRAPH=kg/002_llm"
	@echo "   make download-wikipedia WIKI_COUNT=50          # download 50 articles"

# ═══════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════

## install: Set up the project and install all dependencies
install:
	uv sync --all-extras
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

## experiment: Run an experiment from a YAML config file (EXP relative to experiments/)
experiment:
	uv run python main.py --experiment experiments/$(EXP)/config.yaml

## neo4j-upload: Upload a knowledge_graph.json to Neo4j (always clears first)
neo4j-upload:
	uv run python -c "from polygraph.kg_export.neo4j.upload import upload_graph; upload_graph('experiments/$(GRAPH)/outputs/knowledge_graph.json', clear=True); print('Uploaded experiments/$(GRAPH)/outputs/knowledge_graph.json to Neo4j (cleared first)')"

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
		--language $(WIKI_LANGUAGE) \
		--snapshot $(WIKI_SNAPSHOT) \
		--output $(WIKI_OUTPUT) \
		--max-scan $(WIKI_MAX_SCAN) \
		$(if $(WIKI_SEED),--seed $(WIKI_SEED),) \
		--verbose || true
