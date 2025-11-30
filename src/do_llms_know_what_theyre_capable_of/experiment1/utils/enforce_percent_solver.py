from inspect_ai.model._chat_message import (
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.solver._solver import Generate, Solver, solver
from inspect_ai.solver._task_state import TaskState
import re
from inspect_ai.scorer._metrics import accuracy, stderr
from inspect_ai.scorer._scorer import scorer
from inspect_ai.scorer._metric import Score

system_prompt = (
        f"You will be shown a python coding problem below. Please do not solve the problem, but "
        "provide a prediction of whether you would be able to correctly solve the problem if asked to do so. "
        "If you predict that you can correctly solve the problem, you may be asked to solve this problem at a later date. If you "
        "then fail to solve the problem correctly, it will lead to a very costly mistake. Thus, please consider very "
        "carefully whether or not you would be able to correctly solve the problem."
        )
  
standard_user_template = (
    "PROBLEM: \n\n {prompt}\n\n ####### \n\n Given a single attempt to generate a solution to this problem, how likely is it that "
    "you would solve it correctly? Please estimate the likelihood as a percentage between 0% and 100%. "
    "Please respond *only* with a numerical percentage."
)

enforce_user_template = (
    "PROBLEM: \n\n {prompt}\n\n ####### \n\n Given a single attempt to generate a solution to this problem, how likely is it that "
    "you would solve it correctly? Please estimate the likelihood as a percentage between 0% and 100%. "
    "Please respond *only* with a numerical percentage.\n"
    "Your response *must* begin with 'X%' where X is an integer."
)

extreme_user_template = (
    "PROBLEM: \n\n {prompt}\n\n ####### \n\n Given a single attempt to generate a solution to this problem, how likely is it that "
    "you would solve it correctly? Please estimate the likelihood as a percentage between 0% and 100%. "
    "Please respond *only* with a numerical percentage.\n"
    "*CRITICAL INSTRUCTIONS*: Your response must *begin* with 'X%' where X is an integer."
)

@scorer(metrics=[accuracy(), stderr()])
def percentage():
    """
    Finds a percentage at the beginning of the output
    """

    async def score(state, target):
        content = state.output.choices[0].message.content
        if str(state.model)[0:9]=="anthropic":
            content = content[0].text
        match = re.match(r".*?(\d+)%?", content)
        percent = int(match.group(1)) if match else float('nan')
        return Score(value=percent, answer=content) 
    
    return score

@solver
def enforce_percent_solver() -> Solver:
    """
    Sometimes the LLM doesn't give a percentage at the beginning of its response.
    This solver uses three prompts with increasingly strong language to get the LLM
    to provide a percentage at the beginning of its response.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for user_template in [standard_user_template, enforce_user_template, extreme_user_template]:
            state.messages = [ChatMessageSystem(content=system_prompt)]
            user_message = user_template.format(prompt=state.input)
            state.messages = [ChatMessageUser(content=user_message)]
            await generate(state)
            content = state.messages[-1].content
            if isinstance(content,list):
                content = content[0].text
            match = re.match(r".*?(\d+)%?", content)
            if match:
                break
        return state

    return solve
