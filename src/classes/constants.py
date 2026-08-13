"""Shared regex constants used by the vocabulary and constrained decoder."""
import re

NUMERIC_TOKEN_RE = re.compile(r"^[\d.\-]+$")
NUMBER_PREFIX_RE = re.compile(r"^-?\d*\.?\d*$")
