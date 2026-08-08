*This project has been created as part of the 42 curriculum by diosoare.*

# Call_Me_Maybe

## Description

Call Me Maybe is a function calling project that translates natural language prompts into structured, machine-executable function calls. Given a set of available function definitions and a set of natural language prompts, the goal is to identify the correct function to call and extract its arguments with the correct types, producing valid JSON output for every single prompt.

The program uses a small local LLM (Qwen/Qwen3-0.6B by default) through a provided SDK, and relies on constrained decoding - a token-by-token generation technique that restricts the model's next-token choices to only those that keep the output both syntactically valid JSON and compliant with the expected function schema - to guarantee 100% valid, parseable output even from an unreliable, low-parameter model.

## Resources

**1) How LLMs generate text, token by token**

- OpenAI Cookbook - [https://cookbook.openai.com/](https://cookbook.openai.com/)
- Google AI for Developers (Gemini docs) - [https://ai.google.dev/](https://ai.google.dev/)
- Hugging Face Learn - [https://huggingface.co/learn](https://huggingface.co/learn)

**2) Tokenization (BPE / SentencePiece)**

- OpenAI tiktoken - [https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)
- Google SentencePiece - [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)
- Hugging Face Tokenizers docs - [https://huggingface.co/docs/tokenizers/index](https://huggingface.co/docs/tokenizers/index)
- Microsoft token fundamentals - [https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens)

**3) What function calling means for LLMs**

- OpenAI function calling - [https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
- Anthropic tool use - [https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)
- Google Gemini function calling - [https://ai.google.dev/gemini-api/docs/function-calling](https://ai.google.dev/gemini-api/docs/function-calling)

**4) JSON and JSON Schema validation**

- JSON standard (RFC 8259) - [https://www.rfc-editor.org/rfc/rfc8259](https://www.rfc-editor.org/rfc/rfc8259)
- JSON Schema official docs - [https://json-schema.org/learn/getting-started-step-by-step](https://json-schema.org/learn/getting-started-step-by-step)
- Python json module - [https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html)
- Pydantic docs - [https://docs.pydantic.dev/](https://docs.pydantic.dev/)

**5) Constrained (grammar-guided) decoding**

- OpenAI Structured Outputs - [https://platform.openai.com/docs/guides/structured-outputs](https://platform.openai.com/docs/guides/structured-outputs)
- llama.cpp grammars (GBNF) - [https://github.com/ggerganov/llama.cpp/tree/master/grammars](https://github.com/ggerganov/llama.cpp/tree/master/grammars)
- Hugging Face text generation docs - [https://huggingface.co/docs/transformers/main/en/main_classes/text_generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

**6) Python fundamentals this project enforces**

- Python tutorial - [https://docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/)
- Python typing - [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)
- Python dataclasses - [https://docs.python.org/3/library/dataclasses.html](https://docs.python.org/3/library/dataclasses.html)
- Python exceptions - [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html)

**7) Project tools: uv, Makefile, pytest**

- uv docs (Astral) - [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- GNU Make manual - [https://www.gnu.org/software/make/manual/make.html](https://www.gnu.org/software/make/manual/make.html)
- pytest docs - [https://docs.pytest.org/](https://docs.pytest.org/)
- Python Packaging User Guide (PyPA) - [https://packaging.python.org/](https://packaging.python.org/)
