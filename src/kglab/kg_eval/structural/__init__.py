"""Structural audit of knowledge graphs.

Available methods:
    audit  — Schema compliance, orphan analysis, density, multi-hop connectivity
"""

from .audit import StructuralAuditor

__all__ = ["StructuralAuditor"]
