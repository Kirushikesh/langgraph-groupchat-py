from collections.abc import Callable
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

default_selector_system_prompt = """You are a group chat manager. Your job is to select the next speaker based on the conversation history and the available participants."""


def llm_based_selector(
    model,
    system_prompt: str = default_selector_system_prompt,
    candidate_func: Callable | None = None,
):
    """Create an LLM-based selector function for choosing the next speaker.

    This function creates a selector that uses a language model to intelligently
    choose the next speaker in a group chat based on conversation history and
    available participants.

    Args:
        model: A language model instance that supports structured output
            (e.g., ChatOpenAI with structured output capability).
        system_prompt: System prompt for the selector. Defaults to a standard
            group chat manager prompt.
        candidate_func: Optional function to filter available speakers. Should
            accept state and return a list of allowed role names. If None,
            all participants in state.roles are considered.

    Returns:
        A selector function that accepts state and returns the name of the
        next speaker or "Terminate" to end the conversation.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langgraph_groupchat import llm_based_selector, no_repeat_candidate
        >>>
        >>> model = ChatOpenAI(model="gpt-4o")
        >>>
        >>> # Create a basic selector
        >>> selector = llm_based_selector(model=model)
        >>>
        >>> # Create a selector with custom system prompt
        >>> custom_prompt = '''You are managing a technical support chat.
        ... Select the most appropriate specialist based on the user's question.'''
        >>> selector = llm_based_selector(
        ...     model=model,
        ...     system_prompt=custom_prompt
        ... )
        >>>
        >>> # Create a selector with candidate filtering
        >>> selector = llm_based_selector(
        ...     model=model,
        ...     candidate_func=no_repeat_candidate
        ... )

    """
    speaker_template = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )

    def select_speaker(state):
        messages = state.get("messages", [])
        allowed_roles = (
            state.get("roles", []) if candidate_func is None else candidate_func(state)
        )

        # Create a copy to avoid mutating the original list
        allowed_roles_with_terminate = [*allowed_roles, "Terminate"]

        # Dynamically create an Enum from the allowed roles
        speaker_enum = Enum(
            "SpeakerEnum", {role: role for role in allowed_roles_with_terminate}
        )

        class SelectParticipant(BaseModel):
            """Select the next speaker based on the system prompt and messages."""

            next_speaker: speaker_enum = Field(
                description="The next speaker in the conversation"
            )

        structured_llm = model.with_structured_output(SelectParticipant)
        selector_chain = speaker_template | structured_llm

        result = selector_chain.invoke({"messages": messages})

        return result.next_speaker.value

    return select_speaker
