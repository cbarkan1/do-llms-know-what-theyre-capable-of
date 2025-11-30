"""
My modified big-code-bench, to be more like the "Instruct" version of BigCodeBench
Based on Inspect's BigCodeBench implementation.
"""

import re
from pathlib import Path
from typing import Any
from warnings import warn
from inspect_ai.solver import generate, system_message, prompt_template

from datasets import load_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    mean,
    scorer,
    std,
)
from inspect_ai.solver import TaskState, generate
from inspect_ai.util import ExecResult, sandbox
from inspect_ai.dataset._dataset import MemoryDataset

VERIFY_TIMEOUT = 90

system_prompt = (
    "You are being asked to write a Python function. You will be shown the function signature and docstring describing what the function should "
    "do, what arguments it should take, what it should return, what libraries it should import, and an example usage of the function. "     
)

user_template = (
        "The code below shows the beginning of a python function, with imports, a function signature, and a docstring that describes "
        "what the function should do. Please generate the full code, including the needed import statements and the complete function.\n\n"
        "```python\n{prompt}\n```"
    )

def record_to_sample(record: dict[str, Any]) -> Sample:
    # Get the integer id from the end of the task_id, which is formatted as "BigCodeBench/0"
    task_id = int(record["task_id"].split("/")[-1])
    return Sample(
        id=task_id,
        input=record["complete_prompt"],
        # Include task_id in metadata for use in an error message below.
        metadata={"test": record["test"], "libs": record["libs"], "task_id_metadata": record["task_id"]}, 
    )

def get_dataset(include_ids=None):
    bigcode = load_dataset("bigcode/bigcodebench", split="v0.1.3")
    if include_ids:
        dataset = MemoryDataset(samples=[record_to_sample(record) for record in bigcode if int(record["task_id"].split("/")[-1]) in include_ids])
    else:
        dataset = MemoryDataset(samples=[record_to_sample(record) for record in bigcode])
    return dataset

@task
def bigcodebench(dataset) -> Task:
    """My modiefied Inspect implementation for the BigCodeBench benchmark
    """

    return Task(
        dataset=dataset,
        solver=[
            system_message(system_prompt), 
            prompt_template(user_template),
            generate(),
        ],
        scorer=verify(),
        sandbox=("docker", (Path(__file__).parent / "compose.yaml").as_posix()),
    )


def find_code(completion: str) -> str:
    """
    Removes Markdown formatting around generated code blocks.

    Copied from humaneval
    """
    pattern_1 = re.compile(r"```python\n(.*?)```", re.DOTALL)
    pattern_2 = re.compile(r"```\n(.*?)```", re.DOTALL)
    matches = pattern_1.findall(completion) + pattern_2.findall(completion)
    if matches:
        extracted_answer = matches[0]
    else:
        extracted_answer = completion
    return str(extracted_answer)


@scorer(metrics=[mean(), std()])
def verify() -> Scorer:
    """Runs the provided test cases in a docker container.

    Just looks for a code block, instead of using bigcodebench's
    [code sanitization](https://github.com/bigcode-project/bigcodebench/blob/main/bigcodebench/sanitize.py)
    """

    async def score(state: TaskState, target: Target) -> Score:
        # extract answer from completion and format for verification
        answer = find_code(state.output.completion)
        code = [
            answer,
            "\n",
            state.metadata["test"],
            "\n",
            """
if __name__ == '__main__':
    unittest.main()""",
        ]

        # verify (returns error status if incorrect)
        try:
            result = await sandbox().exec(
                cmd=["python", "-c", "".join(code)],
                timeout=VERIFY_TIMEOUT,
            )
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")

        if not result.success:
            if "ModuleNotFoundError" in result.stderr:
                warn(
                    f"There is a missing module during the docker execution: for task {state.metadata['task_id_metadata']}. The required libraries for this question are {state.metadata['libs']} \n This resulted in {result.stderr}"
                )

        explanation = "".join(
            ["The following verification code was executed:\n\n"]
            + ["```python\n\n"]
            + code
            + ["\n```\n"]
        )
        if result.success:
            explanation = explanation + "The submission was correct."
        else:
            explanation = (
                explanation
                + f"The submission was incorrect with the following error: \n\n{result.stderr}"
            )
        # return score w/ model answer + scoring details
        return Score(
            value=CORRECT if result.success else INCORRECT,
            answer=answer,
            explanation=explanation,
        )

    return score
