import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from .decoder import ConstrainedDecoder
from .models import (
    FunctionCallResult, FunctionDefinition, Small_LLM_Model, Vocabulary
)


class FunctionCallEngine(BaseModel):
    """Loads inputs, runs the generation loop, writes the results file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    definitions_path: Path
    prompts_path: Path
    output_path: Path

    function_definitions: list[FunctionDefinition] = []
    prompts: list[str] = []
    vocabulary: Vocabulary | None = None
    decoder: ConstrainedDecoder | None = None
    results: list[FunctionCallResult] = []

    def load_inputs(self) -> None:
        """Read and validate both input JSON files, then prepare the LLM."""
        self.function_definitions = [
            FunctionDefinition(**item)
            for item in self._read_json_array(self.definitions_path)
        ]
        if not self.function_definitions:
            raise ValueError(
                f"No function definitions found in {self.definitions_path}"
            )
        self.prompts = [
            item["prompt"]
            for item in self._read_json_array(self.prompts_path)
            if isinstance(item, dict) and "prompt" in item
        ]

        self.vocabulary = Vocabulary(llm=self.llm)
        self.vocabulary.build()
        self.decoder = ConstrainedDecoder(
            vocabulary=self.vocabulary,
            function_definitions=self.function_definitions,
        )

    def run(self) -> list[FunctionCallResult]:
        """Process every prompt, skipping ones that fail without crashing."""
        if self.decoder is None:
            raise RuntimeError("load_inputs() must be called before run()")
        self.results = []
        for prompt in self.prompts:
            try:
                self.results.append(self._process_prompt(prompt))
            except Exception as error:
                print(f"Skipping prompt {prompt!r}: {error}")
        return self.results

    def write_output(self) -> None:
        """Serialize the results list as the required output JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.model_dump() for result in self.results]
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _process_prompt(self, prompt: str) -> FunctionCallResult:
        assert self.decoder is not None
        header = (
            f'Request: "{prompt}"\n'
            "Available functions:\n"
            + "\n".join(
                f"- {fd.name}: {fd.description}"
                for fd in self.function_definitions
            )
            + "\nThe function to call is: "
        )
        base_ids = self.llm.encode(header)[0].tolist()
        function_def, ids = self.decoder.select_function_name(
            self.llm, base_ids
        )
        parameters = self.decoder.generate_parameters(
            self.llm, ids, function_def
        )
        return FunctionCallResult(
            prompt=prompt, name=function_def.name, parameters=parameters
        )

    @staticmethod
    def _read_json_array(path: Path) -> list[Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Input file not found: {path}")
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}: {error}")
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data
