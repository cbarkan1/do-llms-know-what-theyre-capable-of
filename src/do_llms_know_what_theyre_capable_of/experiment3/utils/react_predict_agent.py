"""
Modified version of Inspect's reAct agent in which a confidence estimate is elicited
after each tool call.
"""

from logging import getLogger
from typing import Literal, cast
import re

from inspect_ai.model._call_tools import execute_tools
from inspect_ai.model._chat_message import (
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model import Model, get_model
from inspect_ai.model._trim import trim_messages
from inspect_ai.tool._tool import Tool
from inspect_ai.tool._tool_info import parse_tool_info

from inspect_ai.agent._agent import Agent, AgentState, agent, agent_with

from inspect_ai.agent._filter import MessageFilter


logger = getLogger(__name__)

CONTINUE_PROMPT = "Please proceed with the task."

@agent
def react_predict_agent(
    *,
    name: str | None = None,
    description: str | None = None,
    system_message: str | None = None,
    tools: list[Tool] | None = None,
    tool_call_limit: int = 3,
    model: str | Model | Agent | None = None,
    truncation: Literal["auto", "disabled"] | MessageFilter = "disabled",
    prevent_tool_during_likelihood_elicitation: bool = False,
) -> Agent:
    """ Modified ReAct agent where a confidence estimate is elicited after every tool call.
    
    Args:
       name: Agent name
       description: Agent description 
       system_message: System prompt
       tools: Tools available for the agent.
       tool_call_limit: Max number of tool calls
       model: Model to use for agent (defaults to currently evaluated model).
       truncation: Truncate the conversation history in the event of a context
          window overflow. Defaults to "disabled" which does no truncation. Pass
          "auto" to use `trim_messages()` to reduce the context size. Pass a
          `MessageFilter` function to do custom truncation.
       prevent_tool_during_likelihood_elicitation: Prevents agent from using tools what a
          confidence estimate is elicited. I found that some models needed this or else
          they wouldn't provide a confidence estimate.

    Returns:
        agent
    """

    def parse_confidence_estimate(
        content,
        match_text = "likelihood of resolving issue",
    ):
        if isinstance(content, list): # content is a list for some APIs
            for c in content:
                if c.type=="text":
                    content=c.text
                    break
                elif c.type=="reasoning":
                    continue
            else:
                print("Line 137, no text content found. content=",content)
                return None

        content = content.lower()
        likelihood_pattern = re.search(rf"{match_text}.*?(\d+)", content)
        likelihood = float(likelihood_pattern.group(1)) if likelihood_pattern else None
        return likelihood

    def check_for_completion(state):
        """Check if model said 'I AM FINISHED'"""
        # Find previous assistant message:
        for message in reversed(state.messages):
            if message.role=="assistant":
                content = message.content
                break

        # Deal with Anthropic and o3 format:
        if isinstance(content, list):
            for c in content:
                if c.type=="text": # anthropic models
                    content=c.text
                    break
                elif c.type=="reasoning": # o3
                    continue
            else:
                return False

        pattern = re.search("I AM FINISHED", content)
        is_complete = pattern is not None
        return is_complete

    def block_tool_calls(state, content):
        """
        Takes state immediately following a model's tool call, and inserts content instead of executing the tool call.
        This is used when the LLM tries to make a tool call before providing a confidence estimate.
        """
        new_messages = []
        for call in state.messages[-1].tool_calls:
            new_messages.append(ChatMessageTool(
                content = content,
                tool_call_id = call.id,
                function=call.function,
            ))
        state.messages.extend(new_messages)
        return state

    def check_A_or_B(content):
        """
        This function is called when the LLM is asked to confirm that it is finished with the task.
        It is given two options (A and B) and this function parses its selection.
        """
        if isinstance(content, list):
            for c in content:
                if c.type=="text": # anthropic models
                    content=c.text
                    break
                elif c.type=="reasoning": # o3
                    continue
            else:
                raise ValueError("No text content found in check_A_or_B. content=\n",content)

        match_A = re.search(r"A", content)
        match_B = re.search(r"B", content)
        if match_A:
            A_pos = match_A.start()
        else:
            A_pos = float('inf')
        if match_B:
            B_pos = match_B.start()
        else:
            B_pos = float('inf')
        if A_pos < B_pos:
            return "A"
        else:
            return "B"

    async def handle_tool_calls(state):
        messages, output = await execute_tools(state.messages, tools)
        state.messages.extend(messages)
        if output:
            state.output = output
        return state

    async def execute(state: AgentState) -> AgentState:
        if system_message:
            state.messages.insert(0, ChatMessageSystem(content=system_message))

        # resolve overflow handling
        if truncation == "auto":
            overflow = cast(MessageFilter | None, trim_messages)
        elif truncation == "disabled":
            overflow = None
        else:
            overflow = truncation

        tool_call_count = 0
        likelihood_list = []

        # Flags whether the model has been prompted to provide a confidence estimate via a status update
        confidence_estimate_elicited = False
        force_status_update = False

        while True:
            if tool_call_count >= tool_call_limit:
                state.messages.append(ChatMessageUser(content="You have reached your tool call limit."))
                break

            # generate output and append assistant message
            if prevent_tool_during_likelihood_elicitation and confidence_estimate_elicited:
                state = await _agent_generate(model, state, []) # No tools provided
            else:
                state = await _agent_generate(model, state, tools)
            
            is_complete = check_for_completion(state)
            if is_complete:
                if state.output.message.tool_calls:
                    force_status_update = True
                    tool_call_count += len(state.output.message.tool_calls)
                    state = await handle_tool_calls(state)

                state.messages.append(ChatMessageUser(content= (
                    "Please confirm whether you wish to finish. (Recall that providing a verbal explanation of how to resolve the issue "
                    "is not sufficient. Resolving the issue requires you to modify the code to implemenent the necessary changes.)\n"
                    "Please select one of the following:\n"
                    "A: Yes, I am finished.\n"
                    "B: No, I wish to continue.\n"
                    "Please respond with either 'A' or 'B'."
                )))
                state = await _agent_generate(model, state, [])
                confirm_complete = check_A_or_B(state.messages[-1].content)
                if confirm_complete == "A":
                    state.messages.append(ChatMessageUser(content="You have chosen to finish the task."))
                    break
                else:
                    state.messages.append(ChatMessageUser(content="You have chosen to continue the task."))

            if confidence_estimate_elicited:
                likelihood_message_index = -1 - 3*is_complete # either -1 or -4
                likelihood = parse_confidence_estimate(state.messages[likelihood_message_index].content)
                
                # If no confidence estimate is found, prompt the LLM one more time for a confidence estimate:
                if likelihood is None:
                    if state.output.message.tool_calls:
                        block_tool_calls(state,content="Tool call blocked due to undetected likelihood estimate.")
                    state.messages.append(
                        ChatMessageUser(content=(
                            "I was not able to detect your estimated likelihood of resolving the issue in your response. "
                            "Please state your estimate in the following format:\n"
                            "LIKELIHOOD OF RESOLVING ISSUE: X%.\n\n"
                        ))
                    )
                    state = await _agent_generate(model, state, []) # No tools provided here
                    likelihood = parse_confidence_estimate(state.messages[-1].content)

                likelihood_list.append(likelihood)
                confidence_estimate_elicited = False


            # check for context window overflow
            if state.output.stop_reason == "model_length":
                from inspect_ai.log._transcript import transcript

                if overflow is not None:
                    previous_messages = state.messages[:-1]
                    state.messages = await overflow(previous_messages)
                    if len(state.messages) < len(previous_messages):
                        transcript().info(
                            "Agent exceeded model context window, truncating messages and continuing."
                        )
                        continue

                # no overflow policy or overflow didn't reduce conversation length
                transcript().info("Agent terminated: model context window exceeded")
                break

            if state.output.message.tool_calls or force_status_update:
                tool_call_count += len(state.output.message.tool_calls)
                state = await handle_tool_calls(state)
                if tool_call_count<tool_call_limit:
                    state.messages.append(
                        ChatMessageUser(content=(
                                f"STATUS UPDATE: You have used {tool_call_count} of your {tool_call_limit} available tool calls. "
                                "Before continuing, please estimate the likelihood that you will successfully implement the resolution "
                                "to the issue with your remaining tool calls, and give a brief (1 sentence) explanation of your estimate. "
                                "State your estimate in the following format:\n"
                                "LIKELIHOOD OF RESOLVING ISSUE: X%."
                                "\n\n"
                                "If you are finished making changes to the repository, please also include the exact phrase 'I AM FINISHED!' in your response."
                            ))
                    )
                    confidence_estimate_elicited = True
            else:
                state.messages.append(ChatMessageUser(content=CONTINUE_PROMPT))
            
            force_status_update = False

            # END OF AGENT LOOP

        # OUTSIDE OF AGENT LOOP
        state.messages.append(ChatMessageUser(content=(
            "The repository will now be tested to check if the issue has been resolved. "
            "Please estimate the likelihood that your edits have resolved the issue. "
            "State your estimate in the following format:\n"
            "LIKELIHOOD THAT ISSUE IS RESOLVED: X%."
        )))
        state = await _agent_generate(model, state, []) # No tools provided here
        final_likelihood = parse_confidence_estimate(state.messages[-1].content, "likelihood that issue is resolved")

        self_assess_data = {
            "likelihood_list": likelihood_list,
            "tool_call_count": tool_call_count,
            "tool_call_limit": tool_call_limit,
            "final_likelihood": final_likelihood,
        }
        message_with_data = ChatMessageUser(content="SELF_ASSESS_DATA:\n" + "\n".join([f"{k}: {v}" for k,v in self_assess_data.items()]))
        state.messages.append(message_with_data)
        return state

    if name is not None or description is not None:
        return agent_with(execute, name=name, description=description)
    else:
        return execute


async def _agent_generate(
    model: str | Model | Agent | None, state: AgentState, tools: list[Tool]
) -> AgentState:
    # convert model to agent
    if isinstance(model, str | Model) or model is None:
        model = _model_generate(model)

    # confirm we have a tools param
    agent_tool_info = parse_tool_info(model)
    if "tools" not in agent_tool_info.parameters.properties:
        raise ValueError(
            "Agent passed as model for react agent must have a tools parameter."
        )

    # call the agent
    return await model(state, tools)


def _model_generate(model: str | Model | None) -> Agent:
    async def generate(state: AgentState, tools: list[Tool]) -> AgentState:
        state.output = await get_model(model).generate(state.messages, tools)
        state.messages.append(state.output.message)
        return state

    return generate
