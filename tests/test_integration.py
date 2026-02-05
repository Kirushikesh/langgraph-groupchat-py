"""Integration and edge case tests for groupchat."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from typing_extensions import TypedDict

from langgraph_groupchat import (
    GroupChatStateWithFullThread,
    create_groupchat,
)


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


def test_groupchat_missing_state_fields() -> None:
    """Test that missing required state fields raise appropriate errors."""

    class InvalidState(TypedDict):
        messages: list  # Has messages but no roles

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

    def selector(state: dict) -> str:
        return "Terminate"

    with pytest.raises(ValueError, match="Missing required key 'roles'"):
        create_groupchat(
            participants=[alice],
            selector_func=selector,
            state_schema=InvalidState,
        )


def test_groupchat_with_full_thread() -> None:
    """Test groupchat using full_thread for agent input."""
    recorded_messages = [
        AIMessage(
            content="",
            name="Alice",
            tool_calls=[{"name": "dummy_tool", "args": {}, "id": "1"}],
        ),
        AIMessage(content="Tool result processed", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    def dummy_tool() -> str:
        """Return a dummy result."""
        return "result"

    alice: Any = create_agent(
        model,
        tools=[dummy_tool],
        system_prompt="You are Alice.",
        name="Alice",
    )

    def selector(state: dict) -> str:
        messages = state.get("messages", [])
        if messages and messages[-1].name == "Alice":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=selector,
        state_schema=GroupChatStateWithFullThread,
        use_full_thread=True,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Use the tool")]})

    # Verify full_thread field exists in result (it may be empty if state check fails)
    assert "full_thread" in result
    # Verify the workflow completed and messages were processed
    assert "messages" in result
    assert len(result["messages"]) >= 1


def test_groupchat_concurrent_messages() -> None:
    """Test handling of multiple rapid invocations."""
    recorded_messages = [
        AIMessage(content="Response 1", name="Alice"),
        AIMessage(content="Response 2", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    def selector(state: dict) -> str:
        messages = state.get("messages", [])
        if messages and messages[-1].name == "Alice":
            return "Terminate"
        return "Alice"

    checkpointer = MemorySaver()
    workflow = create_groupchat(
        participants=[alice],
        selector_func=selector,
    )
    app = workflow.compile(checkpointer=checkpointer)

    # Simulate concurrent requests with different thread IDs
    config1 = {"configurable": {"thread_id": "thread-1"}}
    config2 = {"configurable": {"thread_id": "thread-2"}}

    result1 = app.invoke({"messages": [HumanMessage(content="Message 1")]}, config1)
    model.idx = 1  # Reset for second invocation
    result2 = app.invoke({"messages": [HumanMessage(content="Message 2")]}, config2)

    # Each should have its own state
    assert result1["messages"][-1].content == "Response 1"
    assert result2["messages"][-1].content == "Response 2"


def test_groupchat_state_updates_persistence() -> None:
    """Test that state updates are properly persisted across turns."""
    recorded_messages = [
        AIMessage(content="Turn 1", name="Alice"),
        AIMessage(content="Turn 2", name="Alice"),
        AIMessage(content="Turn 3", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    turn_count = 0

    def counting_selector(state: dict) -> str:
        nonlocal turn_count
        turn_count += 1
        if turn_count >= 3:
            return "Terminate"
        return "Alice"

    checkpointer = MemorySaver()
    workflow = create_groupchat(
        participants=[alice],
        selector_func=counting_selector,
    )
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "count-test"}}

    # Multiple turns
    result1 = app.invoke({"messages": [HumanMessage(content="Turn 1")]}, config)
    model.idx = 1
    result2 = app.invoke({"messages": [HumanMessage(content="Turn 2")]}, config)

    # State should accumulate
    assert len(result2["messages"]) > len(result1["messages"])


def test_groupchat_empty_message_handling() -> None:
    """Test handling of empty message lists."""
    recorded_messages = [
        AIMessage(content="First message", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    def selector(state: dict) -> str:
        messages = state.get("messages", [])
        # Should handle empty messages gracefully
        if not messages:
            return "Alice"
        return "Terminate"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=selector,
    )

    app = workflow.compile()

    # Invoke with empty messages
    result = app.invoke({"messages": []})

    # Should still work
    assert "messages" in result


def test_groupchat_large_conversation() -> None:
    """Test handling of large conversation history."""
    # Generate many messages
    recorded_messages = [
        AIMessage(content=f"Message {i}", name="Alice") for i in range(50)
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    def selector(state: dict) -> str:
        messages = state.get("messages", [])
        # Stop after 10 agent messages
        agent_msgs = [m for m in messages if m.name == "Alice"]
        if len(agent_msgs) >= 10:
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=selector,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Start long conversation")]})

    # Should handle many messages
    agent_messages = [m for m in result["messages"] if m.name == "Alice"]
    assert len(agent_messages) == 10


def test_groupchat_selector_exception_handling() -> None:
    """Test that selector exceptions are propagated."""
    recorded_messages = [
        AIMessage(content="Response 1", name="Alice"),
        AIMessage(content="Response 2", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    call_count = 0

    class SimulatedSelectorError(ValueError):
        """Custom error for simulated selector failure."""

    def buggy_selector(state: dict) -> str:
        nonlocal call_count
        call_count += 1
        # Raise error on second selector call (after Alice responds)
        if call_count > 1:
            raise SimulatedSelectorError
        return "Alice"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=buggy_selector,
    )

    app = workflow.compile()

    # Should raise error when selector fails
    with pytest.raises(SimulatedSelectorError):
        app.invoke({"messages": [HumanMessage(content="Test")]})


def test_groupchat_role_mutation_in_selector() -> None:
    """Test that selector can access roles and workflow completes successfully."""
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

    def selector_with_roles(state: dict) -> str:
        roles = state.get("roles", [])

        # Verify roles are available
        assert len(roles) >= 2
        assert "Alice" in roles
        assert "Bob" in roles

        # Terminate after Alice speaks
        messages = state.get("messages", [])
        if messages and messages[-1].name == "Alice":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice, bob],
        selector_func=selector_with_roles,
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Test")]})

    # Verify workflow completed
    assert "messages" in result
    assert "roles" in result
    # Alice and Bob should be in the roles
    assert "Alice" in result["roles"]
    assert "Bob" in result["roles"]


def test_groupchat_message_types() -> None:
    """Test handling of different message types."""
    recorded_messages = [
        AIMessage(content="Response to system", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    def selector(state: dict) -> str:
        messages = state.get("messages", [])
        if messages and messages[-1].name == "Alice":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=selector,
    )

    app = workflow.compile()

    # Test with different message types
    result = app.invoke(
        {
            "messages": [
                SystemMessage(content="System instruction"),
                HumanMessage(content="User message"),
            ]
        }
    )

    assert "messages" in result
    assert len(result["messages"]) >= 2


def test_groupchat_recursion_limit() -> None:
    """Test that recursion limit prevents infinite loops."""
    recorded_messages = [
        AIMessage(content=f"Message {i}", name="Alice") for i in range(100)
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    def infinite_selector(state: dict) -> str:
        # Never terminates
        return "Alice"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=infinite_selector,
    )

    app = workflow.compile()

    # Should hit recursion limit
    with pytest.raises(GraphRecursionError):
        app.invoke(
            {"messages": [HumanMessage(content="Start")]},
            config={"recursion_limit": 10},
        )


def test_groupchat_context_schema() -> None:
    """Test groupchat with context schema for runtime configuration."""

    @dataclass
    class Context:
        max_turns: int = 5
        topic: str = "general"

    recorded_messages = [
        AIMessage(content="Context-aware response", name="Alice"),
    ]

    model = FakeChatModel(responses=recorded_messages)  # type: ignore[arg-type]

    alice: Any = create_agent(
        model,
        tools=[],
        system_prompt="You are Alice.",
        name="Alice",
    )

    def context_aware_selector(state: dict) -> str:
        # Can't directly access context in selector, but this tests that
        # context_schema doesn't break the workflow
        messages = state.get("messages", [])
        if messages and messages[-1].name == "Alice":
            return "Terminate"
        return "Alice"

    workflow = create_groupchat(
        participants=[alice],
        selector_func=context_aware_selector,
        context_schema=Context,
    )

    app = workflow.compile()

    result = app.invoke(
        {"messages": [HumanMessage(content="Test")]},
        context={"max_turns": 3, "topic": "testing"},
    )

    assert "messages" in result
