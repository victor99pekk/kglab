SHELL   := /bin/bash

# ═══════════════════════════════════════════════════════════
# Config & Variables
# ═══════════════════════════════════════════════════════════

INPUT      ?= data/wikipedia/
OUTPUT     ?=
VARIANT    ?= surface
NEO4J      ?= 0
CLEAR_NEO4J ?= 0
LINKING    ?= 0
EXP        ?= kg/001_baseline

WIKI_COUNT        ?= 3
WIKI_LANGUAGE     ?= en
WIKI_SNAPSHOT     ?= 20231101
WIKI_OUTPUT       ?= data/wikipedia/random_articles.jsonl
WIKI_MAX_SCAN     ?= 10000
WIKI_SEED         ?=
WIKI_STRATEGY     ?= random
WIKI_TARGET_DEGREE ?= 3.0
WIKI_EXCLUDE_NS   ?= Help:,Template:

# Expands to ", seed=<N>" when WIKI_SEED is set, otherwise to nothing.
# (The comma must live in a variable — it terminates the $(if ...) arguments.)
COMMA := ,
WIKI_SEED_ARG = $(if $(WIKI_SEED),$(COMMA) seed=$(WIKI_SEED),)

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

## help: Show available targets and their descriptions
help:
	@echo "Usage: make <target> [INPUT=...] [OUTPUT=...] [VARIANT=surface|semantic] [EXP=...] [WIKI_COUNT=20]"
	@echo ""
	@echo "── Setup ────────────────────────────────────────────"
	@echo "   install           Sync dependencies with uv + download spaCy model"
	@echo "   clean             Remove generated output folders"
	@echo ""
	@echo "── Data ─────────────────────────────────────────────"
	@echo "   download-wikipedia  Download + enrich Wikipedia articles as JSONL"
	@echo "   enrich-wikipedia    Add outgoing hyperlinks to existing Wikipedia JSONL"
	@echo "   wikipedia-full      Download + enrich Wikipedia articles in one step"
	@echo ""
	@echo "   Wikipedia vars: WIKI_COUNT WIKI_LANGUAGE WIKI_STRATEGY WIKI_TARGET_DEGREE WIKI_OUTPUT"
	@echo "   wikimedia-topic-labels           Prepare pinned Wikimedia topic labels"
	@echo "   wikimedia-topic-labels-validate  Validate label manifest and taxonomy"
	@echo "   wikimedia-topic-labels-download  Download and verify English labels"
	@echo "   wikimedia-topic-labels-prepare   Normalize labels and create QID splits"
	@echo ""
	@echo "── Pipeline ─────────────────────────────────────────"
	@echo "   build-kg          Run the full pipeline (preprocess → build KG → evaluate → export)"
	@echo "   experiment        Run an experiment from a YAML config (set EXP= path)"
	@echo "   neo4j-upload      Upload an existing knowledge_graph.json to Neo4j (clears first)"
	@echo ""
	@echo "── Neo4j ────────────────────────────────────────────"
	@echo "   make build-kg NEO4J=1              # stream the KG directly into Neo4j while building"
	@echo "   make build-kg NEO4J=1 CLEAR_NEO4J=1  # wipe Neo4j first, then stream"
	@echo ""
	@echo "── Pipeline Variants ────────────────────────────────"
	@echo "   surface           Fast: spaCy sm + string matching resolution"
	@echo "   semantic          Deep: spaCy lg + embedding-based resolution"
	@echo ""
	@echo "── Dev ──────────────────────────────────────────────"
	@echo "   test              Run the test suite"
	@echo ""
	@echo "── Quick Start ──────────────────────────────────────"
	@echo "   make install                                   # one-time setup"
	@echo "   make test                                      # verify everything works"
	@echo "   make build-kg                                  # baseline pipeline (default)"
	@echo "   make build-kg VARIANT=semantic                  # semantic pipeline (recommended)"
	@echo "   make build-kg NEO4J=1                           # stream directly into Neo4j"
	@echo "   make experiment                                # baseline experiment (001)"
	@echo "   make neo4j-upload GRAPH_PATH=generated_KGs/KG_0/knowledge_graph.json"
	@echo "   make download-wikipedia WIKI_COUNT=50          # download + enrich 50 articles"
	@echo "   make wikipedia-full WIKI_COUNT=50              # same as download-wikipedia (always enriches)"

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
	@neo4j_flag=""; clear_flag=""; linking_flag=""; output_flag=""; \
	if [ "$(NEO4J)" = "1" ]; then neo4j_flag="--neo4j"; fi; \
	if [ "$(CLEAR_NEO4J)" = "1" ]; then clear_flag="--clear-neo4j"; fi; \
	if [ "$(LINKING)" = "1" ]; then linking_flag="--linking"; fi; \
	if [ -n "$(OUTPUT)" ]; then output_flag="-o $(OUTPUT)"; fi; \
	uv run python main.py -i $(INPUT) $$output_flag --variant $(VARIANT) $$neo4j_flag $$clear_flag $$linking_flag

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

## download-wikipedia: Download Wikipedia articles as Polygraph JSONL
##   make download-wikipedia WIKI_STRATEGY=degree WIKI_COUNT=20 WIKI_TARGET_DEGREE=3.0
##   make download-wikipedia WIKI_EXCLUDE_NS=""  # include all namespaces
download-wikipedia:
	uv run python -c "from polygraph.data import Data, RandomSampler, SpecificSampler, DegreeSampler; \
	sampler = DegreeSampler(count=$(WIKI_COUNT), target_degree=$(WIKI_TARGET_DEGREE), max_scan=$(WIKI_MAX_SCAN)$(WIKI_SEED_ARG), exclude_namespaces=[ns for ns in '$(WIKI_EXCLUDE_NS)'.split(',') if ns]) if '$(WIKI_STRATEGY)' == 'degree' else (SpecificSampler() if '$(WIKI_STRATEGY)' == 'specific' else RandomSampler(count=$(WIKI_COUNT), max_scan=$(WIKI_MAX_SCAN)$(WIKI_SEED_ARG))); \
	Data.download('wikipedia', path='$(WIKI_OUTPUT)', language='$(WIKI_LANGUAGE)', snapshot='$(WIKI_SNAPSHOT)', sampler=sampler)"

## enrich-wikipedia: Add outgoing Wikipedia hyperlinks to existing JSONL
enrich-wikipedia:
	uv run python -c "from polygraph.data import Data; Data.enrich('wikipedia', input_path='$(WIKI_OUTPUT)', language='$(WIKI_LANGUAGE)')"

## download-data: Download Wikipedia data (edit values in tools/data_retrieval/download_data.py)
download-data:
	uv run python tools/data_retrieval/download_data.py

## wikipedia-full: Download + enrich Wikipedia articles (download always enriches)
wikipedia-full: download-wikipedia
	@echo "Done — $(WIKI_OUTPUT) ready with hyperlinks"

## benchmarks-data: Download and cache all benchmark gold datasets
benchmarks-data:
	uv run python -c "
	from polygraph.data import Data
	datasets = [
	    ('bench_ner',        'benchmarks/data/ner_gold.jsonl'),
	    ('bench_dedup',      'benchmarks/data/dedup_gold.jsonl'),
	    ('bench_resolution', 'benchmarks/data/resolution_gold.jsonl'),
	    ('bench_quality',    'benchmarks/data/quality_gold.jsonl'),
	    ('bench_rag',        'benchmarks/data/rag_gold.jsonl'),
	    ('bench_chunking',   'benchmarks/data/chunking_gold.jsonl'),
	]
	for name, path in datasets:
	    Data.download(name, path=path)
	    print(f'  {name} → {path}')
	print('Done — all benchmark datasets cached')
	"

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
