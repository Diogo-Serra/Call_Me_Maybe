import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..llm_sdk.llm_sdk import Small_LLM_Model
from .constants import NUMERIC_TOKEN_RE


class FunctionDefinition(BaseModel):
    """One callable function entry parsed from functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, dict]
    returns: dict


class FunctionCallResult(BaseModel):
    """Resolved function call for a prompt, written to the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]


class Vocabulary(BaseModel):
    """Token-id <-> string map built from the LLM's vocab file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    vocab_path: str | None = None
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}
    numeric_token_ids: list[int] = []

    def build(self) -> None:
        """Download the vocab file and decode it into id -> string form."""
        self.vocab_path = self.llm.get_path_to_vocab_file()
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)

        byte_decoder = self._byte_decoder()
        id_to_token: dict[int, str] = {}
        for raw_token, token_id in self.token_to_id.items():
            raw_bytes = bytes(byte_decoder[c] for c in raw_token)
            id_to_token[token_id] = raw_bytes.decode("utf-8", errors="ignore")
        self.id_to_token = id_to_token
        self.numeric_token_ids = [
            token_id for token_id, text in id_to_token.items()
            if text and NUMERIC_TOKEN_RE.match(text)
        ]

    @staticmethod
    def _byte_decoder() -> dict[str, int]:
        """Reverse of GPT-2's byte-level BPE byte-to-unicode mapping."""
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\xa1"), ord("\xac") + 1))
            + list(range(ord("\xae"), ord("\xff") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        byte_encoder = dict(zip(bs, [chr(c) for c in cs]))
        return {v: k for k, v in byte_encoder.items()}
