SHELL   := /bin/bash

# ═══════════════════════════════════════════════════════════
# Config & Variables
# ═══════════════════════════════════════════════════════════

INPUT      ?= data/wikipedia/
OUTPUT     ?= output/baseline
VARIANT    ?= baseline
NEO4J      ?= 0
CLEAR_NEO4J ?= 0
EXP        ?= kg/001_baseline

WIKI_COUNT    ?= 3
WIKI_LANGUAGE ?= en
WIKI_SNAPSHOT ?= 20231101
WIKI_OUTPUT   ?= data/wikipedia/random_articles.jsonl
WIKI_MAX_SCAN ?= 10000
WIKI_SEED     ?=

GRAPH         ?= kg/001_baseline
GRAPH_PATH    ?=

WIKIMEDIA_TOPIC_EXP       ?= experiments/ML_models/002_wikimedia_article_topic
WIKIMEDIA_TOPIC_TOOL      := tools/data_retrieval/prepare_wikimedia_topic_labels.py
WIKIMEDIA_TOPIC_MANIFEST  := $(WIKIMEDIA_TOPIC_EXP)/dataset_manifest.yaml
WIKIMEDIA_TOPIC_TAXONOMY  := $(WIKIMEDIA_TOPIC_EXP)/wikimedia_topics_64.yaml
WIKIMEDIA_TOPIC_ARTIFACTS := $(WIKIMEDIA_TOPIC_EXP)/artifacts
WIKIMEDIA_TOPIC_LABELS    := $(WIKIMEDIA_TOPIC_ARTIFACTS)/prepared/article_topic_labels.jsonl
WIKIMEDIA_TOPIC_SUMMARY   := $(WIKIMEDIA_TOPIC_ARTIFACTS)/prepared/summary.json

.PHONY: help install clean test build-kg experiment neo4j-upload download-wikipedia \
	enrich-wikipedia wikipedia-full \
	wikimedia-topic-labels wikimedia-topic-labels-validate \
	wikimedia-topic-labels-download wikimedia-topic-labels-prepare

help:
	@echo "Usage: make <target> [INPUT=...] [OUTPUT=...] [VARIANT=baseline] [EXP=...] [WIKI_COUNT=20]"
	@echo ""
	@echo "── Setup ────────────────────────────────────────────"
	@echo "   install           Sync dependencies with uv + download spaCy model"
	@echo "   clean             Remove generated output folders"
	@echo ""
	@echo "── Data ─────────────────────────────────────────────"
	@echo "   download-wikipedia  Download random Wikipedia articles as JSONL"
	@echo "   enrich-wikipedia    Add outgoing hyperlinks to existing Wikipedia JSONL"
	@echo "   wikipedia-full      Download + enrich Wikipedia articles in one step"
	@echo "   wikimedia-topic-labels           Prepare pinned Wikimedia topic labels"
	@echo "   wikimedia-topic-labels-validate  Validate label manifest and taxonomy"
	@echo "   wikimedia-topic-labels-download  Download and verify English labels"
	@echo "   wikimedia-topic-labels-prepare   Normalize labels and create QID splits"
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
	@echo "   make wikipedia-full WIKI_COUNT=50              # download + enrich 50 articles"
	@echo "   make wikimedia-topic-labels                    # Experiment 002 labels"

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
	@neo4j_flag=""; clear_flag=""; \
	if [ "$(NEO4J)" = "1" ]; then neo4j_flag="--neo4j"; fi; \
	if [ "$(CLEAR_NEO4J)" = "1" ]; then clear_flag="--clear-neo4j"; fi; \
	uv run python main.py -i $(INPUT) -o $(OUTPUT) --variant $(VARIANT) $$neo4j_flag $$clear_flag

## experiment: Run an experiment from a YAML config file (EXP relative to experiments/)
experiment:
	uv run python main.py --experiment experiments/$(EXP)/config.yaml

## neo4j-upload: Upload a knowledge_graph.json to Neo4j (always clears first)
##   make neo4j-upload GRAPH=kg/002_llm                  # experiments/kg/002_llm/results/
##   make neo4j-upload GRAPH_PATH=/tmp/polygraph_test/knowledge_graph.json   # arbitrary path
neo4j-upload:
	@if [ -n "$(GRAPH_PATH)" ]; then \
		uv run python -c "from dotenv import load_dotenv; load_dotenv(); from polygraph.kg_export.neo4j.upload import upload_graph; upload_graph('$(GRAPH_PATH)', clear=True); print('Uploaded $(GRAPH_PATH) to Neo4j (cleared first)')"; \
	else \
		uv run python -c "from dotenv import load_dotenv; load_dotenv(); from polygraph.kg_export.neo4j.upload import upload_graph; upload_graph('experiments/$(GRAPH)/results/knowledge_graph.json', clear=True); print('Uploaded experiments/$(GRAPH)/results/knowledge_graph.json to Neo4j (cleared first)')"; \
	fi

# ═══════════════════════════════════════════════════════════
# Dev
# ═══════════════════════════════════════════════════════════

## test: Run the test suite
test:
	uv run pytest tests/ -v

## download-wikipedia: Download random Wikipedia articles as Polygraph JSONL
download-wikipedia:
	uv run python tools/data_retrieval/download_wikipedia_random.py \
		--count $(WIKI_COUNT) \
		--language $(WIKI_LANGUAGE) \
		--snapshot $(WIKI_SNAPSHOT) \
		--output $(WIKI_OUTPUT) \
		--max-scan $(WIKI_MAX_SCAN) \
		$(if $(WIKI_SEED),--seed $(WIKI_SEED),) \
		--verbose || true

## enrich-wikipedia: Add outgoing Wikipedia hyperlinks to existing JSONL
enrich-wikipedia:
	uv run python tools/data_retrieval/enrich_wikipedia_links.py \
		--input $(WIKI_OUTPUT) \
		--output $(WIKI_OUTPUT) \
		--language $(WIKI_LANGUAGE) \
		--verbose

## wikipedia-full: Download random articles + enrich with hyperlinks
wikipedia-full: download-wikipedia enrich-wikipedia
	@echo "Done — $(WIKI_OUTPUT) ready with hyperlinks"

## wikimedia-topic-labels-validate: Validate pinned Wikimedia manifest and taxonomy
wikimedia-topic-labels-validate:
	uv run $(WIKIMEDIA_TOPIC_TOOL) \
		--manifest $(WIKIMEDIA_TOPIC_MANIFEST) \
		--taxonomy $(WIKIMEDIA_TOPIC_TAXONOMY) \
		--artifacts-dir $(WIKIMEDIA_TOPIC_ARTIFACTS) \
		validate

## wikimedia-topic-labels-download: Download and verify pinned English topic labels
wikimedia-topic-labels-download: wikimedia-topic-labels-validate
	uv run $(WIKIMEDIA_TOPIC_TOOL) \
		--manifest $(WIKIMEDIA_TOPIC_MANIFEST) \
		--taxonomy $(WIKIMEDIA_TOPIC_TAXONOMY) \
		--artifacts-dir $(WIKIMEDIA_TOPIC_ARTIFACTS) \
		download labels_en

## wikimedia-topic-labels-prepare: Normalize labels and create deterministic QID splits
wikimedia-topic-labels-prepare: wikimedia-topic-labels-download
	@if test -s $(WIKIMEDIA_TOPIC_LABELS) && test -s $(WIKIMEDIA_TOPIC_SUMMARY); then \
		echo "Wikimedia topic labels already prepared: $(WIKIMEDIA_TOPIC_LABELS)"; \
	else \
		uv run $(WIKIMEDIA_TOPIC_TOOL) \
			--manifest $(WIKIMEDIA_TOPIC_MANIFEST) \
			--taxonomy $(WIKIMEDIA_TOPIC_TAXONOMY) \
			--artifacts-dir $(WIKIMEDIA_TOPIC_ARTIFACTS) \
			prepare; \
	fi

## wikimedia-topic-labels: Complete Experiment 002 label preparation workflow
wikimedia-topic-labels: wikimedia-topic-labels-prepare
