"""Explicit operator registry. Unknown operators always fail validation."""
from .engine import execute, validate_constraint, OPERATOR_FAMILY

__all__ = ["execute", "validate_constraint", "OPERATOR_FAMILY"]
