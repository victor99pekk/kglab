"""Abstract pipeline base class — define stages, inherit to create variants."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from polygraph._shared import Document
from polygraph._shared.stage_config import EvalConfig, ExportConfig


class Pipeline(ABC):
    """Research pipeline with swappable stages.

    Subclass and override any stage method to create a variant. Stages are
    independent — call them individually or chain via run().

    Usage:
        class MyPipeline(Pipeline):
            def preprocess(self) -> list[Document]: ...
            def build_kg(self, chunks) -> dict: ...

        pipe = MyPipeline(input_paths=["data/"], output_dir="output/")
        pipe.execute()              # full pipeline
        # — or —
        result = pipe.preprocess()
        kg = pipe.build_kg(result)
        pipe.evaluate(kg)
        pipe.export(kg)
    """

    def __init__(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        **kwargs: Any,
    ) -> None:
        self.input_paths = [Path(p) for p in input_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._config = kwargs

    # ── Abstract stages (must implement) ────────────────────────

    @abstractmethod
    def preprocess(self) -> list[Document]:
        """Raw files → clean, deduplicated chunks.

        May also return a ``PreprocessResult`` if extraction was performed
        during preprocessing — ``build_kg`` handles both.
        """
        ...

    @abstractmethod
    def build_kg(self, chunks: list[Document] | Any) -> dict[str, Any]:
        """Chunks or PreprocessResult → knowledge graph.

        Returns ``{"graph", "entities", "triples"}``.
        """
        ...

    # ── Default stages (override optional) ──────────────────────

    def evaluate(
        self,
        kg: dict[str, Any],
        llm_client: Any = None,
        config: EvalConfig | None = None,
    ) -> dict[str, Any]:
        """Evaluate KG quality. Default: basic metrics + structural audit.

        Args:
            kg: Dict with ``graph``, ``entities``, ``triples`` keys.
            llm_client: Optional LLM callable for accuracy evaluation.
                When provided, also runs semantic accuracy and triple
                classification checks. Provider-agnostic — any
                ``(prompt: str) -> str`` callable works.
            config: Evaluation configuration. If ``None``, runs quality
                and structural evaluation (the default). Pass
                ``EvalConfig(accuracy_enabled=True)`` to also run
                accuracy checks with an LLM.
        """
        from polygraph.kg_eval import metrics, structural
        from polygraph.kg_eval.metrics import AccuracyEvaluator

        cfg = config or EvalConfig()
        report: dict[str, Any] = {}

        if cfg.quality_enabled:
            report.update(metrics.evaluate(kg["graph"], kg["entities"], kg["triples"]))

        if cfg.structural_enabled:
            report["structural_audit"] = structural.run(kg["graph"], kg["entities"], kg["triples"])

        if cfg.accuracy_enabled and llm_client is not None:
            accuracy_eval = AccuracyEvaluator(llm_client=llm_client)
            report["accuracy"] = accuracy_eval.evaluate(kg["graph"], kg["entities"], kg["triples"])

        path = self.output_dir / "metrics.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        print(f"[evaluate] overall_score={report.get('overall_score', 0):.2f} → {path}")
        return report

    def export(self, kg: dict[str, Any], config: ExportConfig | None = None) -> None:
        """Export KG to configured formats.

        Args:
            kg: Dict with ``graph``, ``entities``, ``triples`` keys.
            config: Export configuration. Defaults to JSON only.
        """
        from polygraph.kg_export import exporter

        cfg = config or ExportConfig()

        for fmt in cfg.formats:
            if fmt == "json":
                exporter.to_json(
                    kg["graph"],
                    kg["entities"],
                    kg["triples"],
                    self.output_dir / "knowledge_graph.json",
                )
            elif fmt == "graphml":
                exporter.to_graphml(
                    kg["graph"],
                    self.output_dir / "knowledge_graph.graphml",
                )
            elif fmt == "neo4j":
                exporter.to_graph_db(
                    self.output_dir / "knowledge_graph.json",
                    backend="neo4j",
                    clear=cfg.neo4j_clear,
                )

        print(f"[export] → {self.output_dir}/")

    def upload_to_graph_db(self, backend: str = "neo4j", clear: bool = False, **kwargs) -> None:
        """Upload the exported KG JSON to a graph database.

        Args:
            backend: Which graph DB to use. Currently only ``"neo4j"`` is supported.
            clear: If True, wipe the database before uploading.
            **kwargs: Forwarded to the backend constructor (e.g. ``uri``, ``user``,
                ``password`` for Neo4j; defaults to ``NEO4J_URI`` / ``NEO4J_USER`` /
                ``NEO4J_PASSWORD`` env vars).

        Example::

            pipe.upload_to_graph_db(backend="neo4j", clear=True)
        """
        from polygraph.kg_export import exporter

        json_path = self.output_dir / "knowledge_graph.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No exported KG found at {json_path}. Run export() first.")
        exporter.to_graph_db(json_path, backend=backend, clear=clear, **kwargs)
        print(f"[{backend}] uploaded → {json_path}")

    def upload_to_neo4j(self, clear: bool = False) -> None:
        """Upload the exported KG JSON to Neo4j (backward-compatible alias)."""
        self.upload_to_graph_db(backend="neo4j", clear=clear)

    def generate_training_data(self, kg: dict[str, Any], chunks: list[Document]) -> None:
        """Generate QA training pairs from KG (and raw chunks as baseline)."""
        from polygraph.finetune.dataset import QADatasetGenerator

        gen = QADatasetGenerator(language="en", seed=42, max_hops=3, test_split=0.2)
        out = self.output_dir / "training_data"
        out.mkdir(parents=True, exist_ok=True)

        kg_qa = gen.generate_from_kg(kg["graph"], kg["entities"], kg["triples"])
        for split_name, pairs in kg_qa.items():
            (out / f"kg_grounded_{split_name}.json").write_text(
                json.dumps(pairs, indent=2, ensure_ascii=False)
            )
        print(f"[training_data] KG-grounded pairs → {out}/")

    # ── Orchestration ───────────────────────────────────────────

    def execute(
        self,
        eval_config: EvalConfig | None = None,
        export_config: ExportConfig | None = None,
    ) -> None:
        """Full pipeline: preprocess → build → evaluate → export.

        Args:
            eval_config: Evaluation configuration (optional).
            export_config: Export configuration (optional).
        """
        print(f"=== {self.__class__.__name__} ===")
        print(f"Input:  {self.input_paths}")
        print(f"Output: {self.output_dir}\n")

        chunks = self.preprocess()
        kg = self.build_kg(chunks)
        self.evaluate(kg, config=eval_config)
        self.export(kg, config=export_config)

        print(f"\nDone — results in {self.output_dir}/")
