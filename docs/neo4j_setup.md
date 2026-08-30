# Neo4j Setup

KGLab streams knowledge graphs into a **disk-backed store** (Neo4j) so huge
KGs never have to fit in Python RAM. This page covers installing and running
Neo4j, setting credentials, and linking it with the `Baseline` pipeline.

## 1. Install the driver

```bash
uv pip install -e ".[neo4j]"   # or: pip install neo4j
```

## 2. Run a Neo4j server (pick one)

**A. Local — Homebrew**

```bash
brew install neo4j
neo4j-admin dbms set-initial-password <your-password>   # first run only
brew services start neo4j
# URI: neo4j://localhost:7687   user: neo4j
```

**B. Local — Docker**

```bash
docker run --name kglab-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your-password> neo4j:5
```

**C. Cloud — Neo4j AuraDB** (free tier)

Create an instance at [console.neo4j.io](https://console.neo4j.io) and copy its
`neo4j+s://<instance-id>.databases.neo4j.io` URI.

## 3. Set credentials

All three are required — the pipeline raises
`RuntimeError("Neo4j credentials not set ...")` otherwise. Add them to `.env`:

```env
NEO4J_URI=neo4j://localhost:7687        # or neo4j+s://... for AuraDB
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

`main.py` loads `.env` automatically. In a notebook, make sure the variables
are set in the kernel's environment before building.

Optional: `NEO4J_TRUST_ALL_CERTIFICATES=false` works around TLS-intercepting
networks (local-only workaround).

## 4. Link with the Baseline pipeline

**A. Simplest — backend flag** (credentials inline, or from `NEO4J_*` env vars
when omitted)

```python
from kglab.pipelines import Baseline

pipe = Baseline(
    input_paths=["data/llm_demo/"],
    output_dir="output/llm_demo/",
    graph_store_backend="neo4j",          # stream into Neo4j — no RAM graph
    graph_store_options={
        "uri": "neo4j://localhost:7687",  # omit to fall back to NEO4J_* env vars
        "user": "neo4j",
        "password": "your-password",
        "clear": True,                    # wipe the DB before each build
    },
)

kg = pipe.execute()
store = kg["graph_store"]                 # Neo4jGraphStore — live Cypher reads
print(store.number_of_nodes(), store.number_of_edges())
print(kg["neo4j_stats"])                  # {'nodes_written': ..., 'edges_written': ...}
```

**B. Pre-configured store** (credentials set once, backend auto-detected)

```python
from kglab.kg_export.graph_store import Neo4jGraphStore
from kglab.pipelines import Baseline

store = Neo4jGraphStore(uri="neo4j://localhost:7687", user="neo4j", password="...")
pipe = Baseline(input_paths=["data/llm_demo/"], output_dir="output/llm_demo/", graph_store=store)
kg = pipe.execute()
assert kg["graph_store"] is store          # same store, reused
```

## What changes vs. the in-memory default

- `kg["graph"]` is `None` — the graph lives in Neo4j, not in Python.
- Reads are live Cypher via the `GraphStore` interface: `number_of_nodes()`,
  `number_of_edges()`, `get_node()`, `successors()`, `predecessors()`,
  `in_degree()`, `nodes()`, `edges()`, ...
- No `knowledge_graph.json` is exported in this mode (the database *is* the
  store). Use `kg["neo4j_stats"]` to confirm what was written.

## Verify & manage

- **Browser:** open http://localhost:7474 (Neo4j Browser), query
  `MATCH (n) RETURN count(n)`.
- **Push an existing JSON export:** `make neo4j-upload GRAPH=kg/002_llm`
  (uploads `experiments/kg/002_llm/results/knowledge_graph.json`, clearing first).
- **Wipe the DB:** pass `"clear": True` in `graph_store_options`, or run
  `MATCH (n) DETACH DELETE n`.

## Related

- End-to-end example: `tutorial/kg_for_llm_training.ipynb`
- Neo4j export/upload docs: `docs/usage.md` → "Neo4j Export"
