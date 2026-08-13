from .constants import NUMBER_PREFIX_RE
from pydantic import BaseModel, ConfigDict
from .models import FunctionDefinition, Small_LLM_Model, Vocabulary


class ConstrainedDecoder(BaseModel):
    """Masks next-token logits so only schema-valid continuations survive."""

    vocabulary: Vocabulary
    function_definitions: list[FunctionDefinition]
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
            logits = llm.get_logits_from_input_ids(ids)
            best_id, best_text, best_logit = None, None, float("-inf")
            for token_id, token_text in self.vocabulary.id_to_token.items():
                if not token_text:
                    continue
                candidate = partial + token_text
                if any(c.startswith(candidate) for c in remaining):
                    if logits[token_id] > best_logit:
                        best_id = token_id
                        best_text = token_text
                        best_logit = logits[token_id]
            if best_id is None:
                break
            ids.append(best_id)
            partial += best_text
        return partial, ids

    def _generate_number(
        self, llm: Small_LLM_Model, input_ids: list[int]
    ) -> tuple[float, list[int]]:
        """Greedily extend a numeric literal, stopping when the model's own
        top choice would break the number format."""
        ids = list(input_ids)
        partial = ""
        for _ in range(12):
            logits = llm.get_logits_from_input_ids(ids)
            top_id_all = max(range(len(logits)), key=lambda i: logits[i])
            top_text_all = self.vocabulary.id_to_token.get(top_id_all, "")
            if partial and not NUMBER_PREFIX_RE.match(partial + top_text_all):
                break
            best_id, best_text, best_logit = None, None, float("-inf")
            for token_id in self.vocabulary.numeric_token_ids:
                token_text = self.vocabulary.id_to_token[token_id]
                if (
                    NUMBER_PREFIX_RE.match(partial + token_text)
                    and logits[token_id] > best_logit
                ):
                    best_id = token_id
                    best_text = token_text
                    best_logit = logits[token_id]
            if best_id is None:
                break
            ids.append(best_id)
            partial += best_text
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
        for _ in range(24):
            logits = llm.get_logits_from_input_ids(ids)
            top_id_all = max(range(len(logits)), key=lambda i: logits[i])
            top_text_all = self.vocabulary.id_to_token.get(top_id_all, "")
            if partial and ("\n" in top_text_all or '"' in top_text_all):
                break
            best_id, best_text, best_logit = None, None, float("-inf")
            for token_id, token_text in self.vocabulary.id_to_token.items():
                if not token_text or "\n" in token_text or '"' in token_text:
                    continue
                if logits[token_id] > best_logit:
                    best_id = token_id
                    best_text = token_text
                    best_logit = logits[token_id]
            if best_id is None:
                break
            ids.append(best_id)
            partial += best_text
        return partial.strip(" '\""), ids
