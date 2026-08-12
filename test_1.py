from os import environ
from src import llm_sdk
from dotenv import load_dotenv

# Look for HuggingFace token
load_dotenv()
HF_TOKEN = environ.get("HF_TOKEN")


# Test 1
def main():
    llm = llm_sdk.Small_LLM_Model()
    print(f"Local LLM model name: {llm._model_name}")
    print("Dependencies: Ok")


if __name__ == "__main__":
    main()
