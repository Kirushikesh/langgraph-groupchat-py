"""Tests for groupchat functionality."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from langgraph_groupchat import GroupChatState, create_groupchat

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class FakeChatModel(BaseChatModel):
    """Fake chat model for testing that returns predefined responses."""

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


def test_basic_groupchat() -> None:
    """Test basic group chat with alternating speakers."""
    # Each agent invocation needs one response from the model
    recorded_messages = [
        AIMessage(content="Hi, I'm Alice!", name="Alice"),
        AIMessage(content="Hello, I'm Bob!", name="Bob"),
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

    # Selector that terminates after Bob speaks
    def selector(state):
        messages = state.get("messages", [])
        if not messages:
            return "Alice"
        last_speaker = messages[-1].name if messages[-1].name else None
        if last_speaker == "Alice":
            return "Bob"
        if last_speaker == "Bob":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Hello everyone!")]})

    # Verify results: Initial + Alice + Bob
    assert len(result["messages"]) == 3
    assert result["messages"][0].content == "Hello everyone!"
    assert result["messages"][1].content == "Hi, I'm Alice!"
    assert result["messages"][2].content == "Hello, I'm Bob!"


def test_groupchat_with_termination() -> None:
    """Test group chat that terminates after specific conditions."""
    recorded_messages = [
        AIMessage(content="Starting discussion", name="Alice"),
        AIMessage(content="Adding my thoughts", name="Bob"),
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

    # Selector that terminates after Bob speaks
    def selector(state):
        messages = state.get("messages", [])
        if not messages:
            return "Alice"
        last_speaker = messages[-1].name if messages[-1].name else None
        if last_speaker == "Alice":
            return "Bob"
        if last_speaker == "Bob":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Start discussion")]})

    # Verify termination
    assert len(result["messages"]) == 3  # Initial + Alice + Bob
    assert result["messages"][-1].name == "Bob"


def test_groupchat_with_checkpointer() -> None:
    """Test group chat with state persistence."""
    recorded_messages = [
        AIMessage(content="First message", name="Alice"),
        AIMessage(content="Second message", name="Bob"),
        AIMessage(content="Third message", name="Alice"),
        AIMessage(content="Fourth message", name="Bob"),
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

    def selector(state):
        messages = state.get("messages", [])
        if not messages:
            return "Alice"
        last_speaker = messages[-1].name if messages[-1].name else None
        if last_speaker == "Alice":
            return "Bob"
        if last_speaker == "Bob":
            return "Terminate"
        return "Alice"

    checkpointer = MemorySaver()
    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=selector,
    )
    app = workflow.compile(checkpointer=checkpointer)

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    # First turn
    turn_1 = app.invoke(
        {"messages": [HumanMessage(content="Hello")]},
        config,
    )

    assert len(turn_1["messages"]) == 3  # Initial + Alice + Bob
    assert turn_1["messages"][-1].name == "Bob"

    # Second turn - should continue from previous state
    turn_2 = app.invoke(
        {"messages": [HumanMessage(content="Continue")]},
        config,
    )

    # Should have all messages from both turns
    assert len(turn_2["messages"]) > 3


def test_groupchat_roles_initialization() -> None:
    """Test that roles are properly initialized even when not provided by user."""
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

    # Selector that uses roles and terminates after Alice
    def selector(state):
        roles = state.get("roles", [])
        messages = state.get("messages", [])
        # This should work even though user didn't pass roles
        assert len(roles) == 2
        assert "Alice" in roles
        assert "Bob" in roles
        # Terminate after Alice speaks
        if messages and messages[-1].name == "Alice":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=selector,
    )

    app = workflow.compile()

    # User doesn't pass roles - they should be auto-initialized
    result = app.invoke({"messages": [HumanMessage(content="Test")]})

    # Verify roles were initialized
    assert "roles" in result
    assert len(result["roles"]) == 2
    assert "Alice" in result["roles"]
    assert "Bob" in result["roles"]


def test_groupchat_with_tools() -> None:
    """Test group chat where agents use tools."""
    # Tool call followed by final response
    recorded_messages = [
        AIMessage(
            content="",
            name="Alice",
            tool_calls=[
                {
                    "name": "add",
                    "args": {"a": 5, "b": 3},
                    "id": "call_123",
                }
            ],
        ),
        AIMessage(content="The sum is 8", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    alice: Any = create_agent(
        model,
        tools=[add],
        system_prompt="You are Alice, a math expert.",
        name="Alice",
    )

    def selector(state):
        messages = state.get("messages", [])
        if not messages:
            return "Alice"
        # Terminate after Alice responds
        if messages[-1].name == "Alice":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="What's 5 + 3?")]})

    # Verify tool was used and response generated
    assert len(result["messages"]) >= 2
    assert result["messages"][-1].content == "The sum is 8"


def test_groupchat_three_participants() -> None:
    """Test group chat with three participants."""
    recorded_messages = [
        AIMessage(content="Alice here", name="Alice"),
        AIMessage(content="Bob speaking", name="Bob"),
        AIMessage(content="Charlie's turn", name="Charlie"),
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

    charlie: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Charlie.",
        name="Charlie",
    )

    # Round-robin selector
    def selector(state):
        messages = state.get("messages", [])
        roles = state.get("roles", [])
        if not messages:
            return roles[0] if roles else "Alice"

        last_speaker = messages[-1].name if messages[-1].name else None
        if last_speaker == "Alice":
            return "Bob"
        if last_speaker == "Bob":
            return "Charlie"
        if last_speaker == "Charlie":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice, bob, charlie],
        selector_func=selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Hello all")]})

    # Verify all three participated
    assert len(result["messages"]) == 4  # Initial + 3 agents
    assert any(m.name == "Alice" for m in result["messages"])
    assert any(m.name == "Bob" for m in result["messages"])
    assert any(m.name == "Charlie" for m in result["messages"])
    assert len(result["roles"]) == 3


def test_groupchat_custom_state() -> None:
    """Test group chat with custom state schema."""

    @dataclass
    class CustomGroupChatState(GroupChatState):
        """Custom state with additional fields."""

        turn_count: int = 0
        topic: str = ""

    recorded_messages = [
        AIMessage(content="Discussing topic", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    def selector(state):
        # Access custom state fields
        turn_count = state.get("turn_count", 0)
        topic = state.get("topic", "")
        assert isinstance(turn_count, int)
        assert isinstance(topic, str)
        # Terminate after first response
        messages = state.get("messages", [])
        if messages and messages[-1].name == "Alice":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=selector,
        state_schema=CustomGroupChatState,
    )

    app = workflow.compile()

    result = app.invoke(
        {
            "messages": [HumanMessage(content="Test")],
            "turn_count": 1,
            "topic": "Testing",
        }
    )

    # Verify custom fields preserved
    assert result["turn_count"] == 1
    assert result["topic"] == "Testing"


def test_groupchat_invalid_speaker() -> None:
    """Test that invalid speaker selection is handled gracefully."""
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

    # Selector that returns invalid speaker name
    def bad_selector(state):
        return "NonExistentAgent"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=bad_selector,
    )

    app = workflow.compile()

    # Should terminate gracefully instead of crashing
    result = app.invoke({"messages": [HumanMessage(content="Test")]})

    # Graph should terminate without error
    assert "messages" in result


def test_groupchat_empty_participants() -> None:
    """Test that empty participants list raises error."""

    def selector(state):
        return "Terminate"

    with pytest.raises(ValueError, match="participants list cannot be empty"):
        create_groupchat(
            participants=[],
            selector_func=selector,
        )
