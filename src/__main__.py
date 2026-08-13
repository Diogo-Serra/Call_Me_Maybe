import sys

try:
    from .classes import Init
    from .llm_sdk import llm_sdk
    from pydantic import ValidationError
    from .classes import FunctionCallEngine
except KeyboardInterrupt:
    print("\nAborted while loading dependencies.")
    sys.exit(130)


def main() -> int:
    config = Init()
    args = config.parse_args()
    definitions_path, prompts_path, output_path = config.resolve_paths(args)

    print(f"\n{4 * '-'} Call_me_maybe project for 42 Lisbon {4 * '-'}\n")
    try:
        print("Preparing LLM model and resolving prompts...")
        llm = llm_sdk.Small_LLM_Model()
        engine = FunctionCallEngine(
            llm=llm,
            definitions_path=definitions_path,
            prompts_path=prompts_path,
            output_path=output_path,
        )
        engine.load_inputs()
        engine.run()
        engine.write_output()
    except ValidationError as error:
        print(f"Error: invalid data in {definitions_path}:")
        for issue in error.errors():
            location = " -> ".join(str(part) for part in issue["loc"])
            print(f"  - {location or '<root>'}: {issue['msg']}")
        return 1
    except (OSError, ValueError) as error:
        print(f"\nError: {error}")
        return 1
    except Exception as error:
        print(f"\nUnexpected error ({type(error).__name__}): {error}")
        return 1
    except (BaseException, KeyboardInterrupt) as error:
        print(f"{error}\nExiting...")
        return 0

    print(f"\nWrote {len(engine.results)} result(s) to:\n{output_path}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
