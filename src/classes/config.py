import argparse
from os import environ
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

# config.py lives in src/classes/, so data/ is one level up, under src/
_SRC_DIR = Path(__file__).parent.parent


class CliArgs(BaseModel):
    """The three optional CLI flags the program accepts."""

    functions_definition: Path | None
    input: Path | None
    output: Path | None


class Init(BaseModel):
    """Loads environment config, default paths, and CLI arguments."""

    hf_token: str | None = None
    data_dir: Path = _SRC_DIR / "data" / "input"
    output_file: Path = (
        _SRC_DIR / "data" / "output" / "function_calling_results.json"
    )

    def model_post_init(self, __context: Any) -> None:
        """Load .env and fill in the HF token once fields are set."""
        load_dotenv()
        if self.hf_token is None:
            self.hf_token = environ.get("HF_TOKEN")

    def parse_args(self) -> CliArgs:
        """Parse the --functions_definition/--input/--output CLI flags."""
        parser = argparse.ArgumentParser(
            prog="python -m src",
            description=(
                "Translate natural language prompts into function calls."
            ),
        )
        parser.add_argument(
            "--functions_definition", type=Path, default=None
        )
        parser.add_argument("--input", type=Path, default=None)
        parser.add_argument("--output", type=Path, default=None)
        namespace = parser.parse_args()
        return CliArgs(
            functions_definition=namespace.functions_definition,
            input=namespace.input,
            output=namespace.output,
        )

    def resolve_paths(
        self, args: CliArgs
    ) -> tuple[Path, Path, Path]:
        """Fall back to the default data paths for any unset CLI flag."""
        definitions_path = args.functions_definition or (
            self.data_dir / "functions_definition.json"
        )
        prompts_path = args.input or (
            self.data_dir / "function_calling_tests.json"
        )
        output_path = args.output or self.output_file
        return definitions_path, prompts_path, output_path
