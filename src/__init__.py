"""Call Me Maybe: schema-constrained function calling with a local LLM.

The package translates natural-language requests into validated function calls.
It uses greedy, token-level constrained decoding to select a declared function
and generate schema-compatible arguments, then serializes the result as JSON.

The package-level API exposes the application configuration, bundled data, and
the provided local LLM SDK used by the command-line pipeline.

run:
    uv run python -m src \
  --functions_definition src/data/input/functions_definition.json \
  --input src/data/input/function_calling_tests.json \
  --output src/data/output/function_calling_results.json
"""

import sys

try:
    from .classes import Init
    from .llm_sdk import llm_sdk
    from .data import input as data_input
except KeyboardInterrupt:
    print("\nAborted while loading dependencies.")
    sys.exit(130)

__all__ = ["llm_sdk", "data_input", "Init"]
