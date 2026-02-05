"""Tests for groupchat selector functions."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph_groupchat import create_groupchat, llm_based_selector


class FakeChatModel(BaseChatModel):
    """Fake chat model for testing."""

    idx: int = 0
    responses: list[BaseMessage]

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        generation = ChatGeneration(message=self.responses[self.idx])
        self.idx += 1
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "FakeChatModel":
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Mock structured output for testing."""

        # Return a callable that returns the next speaker
        def structured_llm(input_value: Any) -> Any:
            # Handle ChatPromptValue - extract messages from it
            if hasattr(input_value, "messages"):
                messages = input_value.messages
            elif isinstance(input_value, dict):
                messages = input_value.get("messages", [])
            else:
                messages = []

            class MockResponse:
                def __init__(self, speaker: str) -> None:
                    self.next_speaker = type("Enum", (), {"value": speaker})()

            if not messages:
                return MockResponse("Alice")

            # Get last speaker
            last_msg = messages[-1] if messages else None
            last_speaker = (
                last_msg.name if last_msg and hasattr(last_msg, "name") else None
            )

            if last_speaker == "Alice":
                return MockResponse("Bob")
            if last_speaker == "Bob":
                return MockResponse("Terminate")
            return MockResponse("Alice")

        return structured_llm


def test_llm_based_selector_basic() -> None:
    """Test LLM-based selector with basic functionality."""
    recorded_messages = [
        AIMessage(content="Alice response", name="Alice"),
        AIMessage(content="Bob response", name="Bob"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    bob: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Bob.",
        name="Bob",
    )

    # Create LLM-based selector
    selector = llm_based_selector(model=model)

    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Hello")]})

    # Verify it works
    assert len(result["messages"]) >= 2
    assert "roles" in result


def test_llm_based_selector_with_custom_prompt() -> None:
    """Test LLM-based selector with custom system prompt."""
    recorded_messages = [
        AIMessage(content="Expert response", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice, an expert.",
        name="Alice",
    )

    custom_prompt = "You are a specialized selector for technical discussions."

    selector = llm_based_selector(
        model=model,
        system_prompt=custom_prompt,
    )

    workflow = create_groupchat(
        participants=[alice],
        selector_func=selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Technical question")]})

    assert "messages" in result


def test_llm_based_selector_with_candidate_func() -> None:
    """Test LLM-based selector with candidate filtering function."""
    recorded_messages = [
        AIMessage(content="Alice speaks", name="Alice"),
        AIMessage(content="Bob speaks", name="Bob"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    bob: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Bob.",
        name="Bob",
    )

    # Candidate function that prevents same speaker twice in a row
    def no_repeat_candidate(state: dict) -> list:
        messages = state.get("messages", [])
        roles = state.get("roles", [])

        if not messages or not roles:
            return roles

        last_speaker = messages[-1].name if messages[-1].name else None
        return [role for role in roles if role != last_speaker]

    selector = llm_based_selector(
        model=model,
        candidate_func=no_repeat_candidate,
    )

    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Start")]})

    # Verify it runs without error
    assert "messages" in result


def test_simple_selector_state_access() -> None:
    """Test that selector can access all state fields."""
    recorded_messages = [
        AIMessage(content="Response", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    bob: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Bob.",
        name="Bob",
    )

    # Selector that accesses different state fields
    def state_aware_selector(state: dict) -> str:
        # Should be able to access messages
        messages = state.get("messages", [])
        assert isinstance(messages, list)

        # Should be able to access roles
        roles = state.get("roles", [])
        assert len(roles) == 2

        # Should work with dict-like access
        assert "messages" in state
        assert "roles" in state

        # Terminate after Alice speaks
        if messages and messages[-1].name == "Alice":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=state_aware_selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Test")]})

    assert "messages" in result
    assert "roles" in result


def test_selector_returns_terminate() -> None:
    """Test selector that returns Terminate to end conversation."""
    recorded_messages = [
        AIMessage(content="Final message", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    # Selector that terminates after first message
    def immediate_terminate_selector(state: dict) -> str:
        messages = state.get("messages", [])
        if len(messages) > 1:  # User message + 1 agent response
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=immediate_terminate_selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Quick task")]})

    # Should terminate after one agent response
    assert len(result["messages"]) == 2  # User message + Alice response


def test_selector_with_message_count_limit() -> None:
    """Test selector that limits conversation length."""
    recorded_messages = [
        AIMessage(content="Message 1", name="Alice"),
        AIMessage(content="Message 2", name="Bob"),
        AIMessage(content="Message 3", name="Alice"),
        AIMessage(content="Message 4", name="Bob"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    bob: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Bob.",
        name="Bob",
    )

    max_turns = 3

    def limited_turn_selector(state: dict) -> str:
        messages = state.get("messages", [])

        # Terminate after max_turns agent messages
        agent_messages = [m for m in messages if m.name in ["Alice", "Bob"]]
        if len(agent_messages) >= max_turns:
            return "Terminate"

        # Alternate between Alice and Bob
        last_speaker = messages[-1].name if messages and messages[-1].name else None
        if last_speaker == "Alice":
            return "Bob"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=limited_turn_selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Start conversation")]})

    # Count agent messages
    agent_messages = [m for m in result["messages"] if m.name in ["Alice", "Bob"]]
    assert len(agent_messages) <= max_turns


def test_selector_priority_based() -> None:
    """Test selector that chooses based on priority/expertise."""
    recorded_messages = [
        AIMessage(content="Math answer", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice, a math expert.",
        name="Alice",
    )

    bob: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Bob, a history expert.",
        name="Bob",
    )

    def priority_selector(state: dict) -> str:
        messages = state.get("messages", [])

        if not messages:
            return "Alice"

        # Terminate after first agent response
        if messages[-1].name in ["Alice", "Bob"]:
            return "Terminate"

        # Simple keyword-based routing
        last_user_message = messages[0].content.lower()

        if "math" in last_user_message or "calculate" in last_user_message:
            return "Alice"
        if "history" in last_user_message or "past" in last_user_message:
            return "Bob"
        return "Terminate"

    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=priority_selector,
    )

    app = workflow.compile()

    # Test math question routes to Alice
    result = app.invoke({"messages": [HumanMessage(content="Calculate 5 + 3")]})

    agent_messages = [m for m in result["messages"] if m.name]
    assert any(m.name == "Alice" for m in agent_messages)
