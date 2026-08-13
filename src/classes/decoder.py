import numpy
from .constants import NUMBER_PREFIX_RE
from pydantic import BaseModel, ConfigDict, PrivateAttr
from .models import FunctionDefinition, Small_LLM_Model, Vocabulary


class ConstrainedDecoder(BaseModel):
    """Masks next-token logits so only schema-valid continuations survive."""

    vocabulary: Vocabulary
    function_definitions: list[FunctionDefinition]
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _string_safe_ids: numpy.ndarray | None = PrivateAttr(default=None)

    def select_function_name(
        self, llm: Small_LLM_Model, input_ids: list[int]
    ) -> tuple[FunctionDefinition, list[int]]:
        """Constrained-decode the name field to one of the known functions."""
        names = [fd.name for fd in self.function_definitions]
        partial, ids = self._generate_enum(llm, input_ids, names)
        for fd in self.function_definitions:
            if fd.name == partial:
                return fd, ids
        for fd in self.function_definitions:
            if fd.name.startswith(partial):
                return fd, ids
        return self.function_definitions[0], ids

    def generate_parameters(
        self,
        llm: Small_LLM_Model,
        input_ids: list[int],
        function_def: FunctionDefinition,
    ) -> dict:
        """Constrained-decode every parameter value per its declared type."""
        parameters: dict[str, object] = {}
        ids = list(input_ids)
        for param_name, param_schema in function_def.parameters.items():
            param_type = param_schema.get("type", "string")
            prompt = f'\nValue for parameter "{param_name}" ({param_type}): '
            ids = ids + llm.encode(prompt)[0].tolist()
            if param_type == "number":
                value, ids = self._generate_number(llm, ids)
            elif param_type == "boolean":
                text, ids = self._generate_enum(llm, ids, ["true", "false"])
                value = text == "true"
            else:
                value, ids = self._generate_string(llm, ids)
            parameters[param_name] = value
        return parameters

    def _generate_enum(
        self,
        llm: Small_LLM_Model,
        input_ids: list[int],
        candidates: list[str],
    ) -> tuple[str, list[int]]:
        """Greedily walk the token trie of the allowed literal strings."""
        ids = list(input_ids)
        partial = ""
        max_steps = max(len(c) for c in candidates) + 2
        for _ in range(max_steps):
            if partial in candidates:
                break
            remaining = [c for c in candidates if c.startswith(partial)]
            if not remaining:
                break
            logits = numpy.asarray(
                llm.get_logits_from_input_ids(ids), dtype=float
            )
            masked = numpy.full(logits.shape, float("-inf"))
            for token_id, token_text in self.vocabulary.id_to_token.items():
                if token_text and any(
                    c.startswith(partial + token_text) for c in remaining
                ):
                    masked[token_id] = logits[token_id]
            if not numpy.isfinite(masked).any():
                break
            best_id = int(numpy.argmax(masked))
            ids.append(best_id)
            partial += self.vocabulary.id_to_token[best_id]
        return partial, ids

    def _generate_number(
        self, llm: Small_LLM_Model, input_ids: list[int]
    ) -> tuple[float, list[int]]:
        """Greedily extend a numeric literal, stopping when the model's own
        top choice would break the number format."""
        ids = list(input_ids)
        partial = ""
        for _ in range(12):
            logits = numpy.asarray(
                llm.get_logits_from_input_ids(ids), dtype=float
            )
            top_id_all = int(numpy.argmax(logits))
            top_text_all = self.vocabulary.id_to_token.get(top_id_all, "")
            if partial and not NUMBER_PREFIX_RE.match(partial + top_text_all):
                break
            masked = numpy.full(logits.shape, float("-inf"))
            for token_id in self.vocabulary.numeric_token_ids:
                token_text = self.vocabulary.id_to_token[token_id]
                if NUMBER_PREFIX_RE.match(partial + token_text):
                    masked[token_id] = logits[token_id]
            if not numpy.isfinite(masked).any():
                break
            best_id = int(numpy.argmax(masked))
            ids.append(best_id)
            partial += self.vocabulary.id_to_token[best_id]
        try:
            return float(partial), ids
        except ValueError:
            return 0.0, ids

    def _generate_string(
        self, llm: Small_LLM_Model, input_ids: list[int]
    ) -> tuple[str, list[int]]:
        """Greedily extend free text, stopping at a quote or newline."""
        ids = list(input_ids)
        partial = ""
        safe_ids = self._get_string_safe_ids()
        for _ in range(24):
            logits = numpy.asarray(
                llm.get_logits_from_input_ids(ids), dtype=float
            )
            top_id_all = int(numpy.argmax(logits))
            top_text_all = self.vocabulary.id_to_token.get(top_id_all, "")
            if partial and ("\n" in top_text_all or '"' in top_text_all):
                break
            masked = numpy.full(logits.shape, float("-inf"))
            masked[safe_ids] = logits[safe_ids]
            if not numpy.isfinite(masked).any():
                break
            best_id = int(numpy.argmax(masked))
            ids.append(best_id)
            partial += self.vocabulary.id_to_token[best_id]
        return partial.strip(" '\""), ids

    def _get_string_safe_ids(self) -> numpy.ndarray:
        """Cache the token ids that never break a JSON string boundary.
        Unlike enum/number continuation, string safety never depends on
        `partial`, so the mask can be built once and reused every step."""
        if self._string_safe_ids is None:
            self._string_safe_ids = numpy.array(
                [
                    token_id
                    for token_id, text in self.vocabulary.id_to_token.items()
                    if text and "\n" not in text and '"' not in text
                ],
                dtype=int,
            )
        return self._string_safe_ids
