import sys
from os import system
from .classes import Init
from .llm_sdk import llm_sdk
from rich.console import Console
from pydantic import ValidationError
from .classes import FunctionCallEngine


def main() -> int:

    config = Init()
    console = Console()
    args = config.parse_args()
    definitions_path, prompts_path, output_path = config.resolve_paths(args)

    try:
        system("clear")
        print(f"\n{4 * '-'} Call_me_maybe project for 42 Lisbon {4 * '-'}\n")

        print("Preparing LLM model and resolving prompts...")
        llm = llm_sdk.Small_LLM_Model()
        engine = FunctionCallEngine(
            llm=llm,
            definitions_path=definitions_path,
            prompts_path=prompts_path,
            output_path=output_path,
        )
        engine.load_inputs()
        with console.status("Generating output results..."):
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

    system("clear")
    total = len(engine.prompts)
    succeeded = len(engine.results)
    if succeeded == total:
        print(
            f"Success!\nWrote {succeeded} result(s) from {total} "
            f"prompt(s) to:\n{output_path}"
        )
    else:
        print(
            f"Partial success: wrote {succeeded}/{total} result(s) "
            f"({total - succeeded} prompt(s) skipped, see messages above) "
            f"to:\n{output_path}"
        )
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
