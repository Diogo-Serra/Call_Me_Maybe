from src import llm_sdk


def main():
    llm = llm_sdk.Small_LLM_Model()
    print("Ok")
    print(llm._model_name)


if __name__ == "__main__":
    main()
