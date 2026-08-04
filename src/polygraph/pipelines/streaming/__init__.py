"""Streaming pipelines — build directly into Neo4j to avoid OOM."""

from polygraph.pipelines.streaming._streaming import StreamingPipeline

__all__ = ["StreamingPipeline"]
