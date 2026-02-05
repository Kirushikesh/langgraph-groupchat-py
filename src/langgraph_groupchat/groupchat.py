import logging
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langchain.messages import AnyMessage
from langchain_core.messages import HumanMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

logger = logging.getLogger(__name__)


@dataclass
class GroupChatState(MessagesState):
    """State schema for group chat extending MessagesState.

    MessagesState already provides:
        messages: Annotated[list[AnyMessage], add_messages]

    We add:
        roles: List of available participant names in the group chat.
    """

    roles: list[str] = field(default_factory=list)


@dataclass
class GroupChatStateWithFullThread(MessagesState):
    """State schema with both filtered messages and full thread.

    Attributes:
        messages: (from MessagesState) Main conversation history
        full_thread: Complete history including all agent interactions
        roles: Available participant names

    """

    full_thread: Annotated[list[AnyMessage], operator.add] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)


def agent_node(agent, *, use_full_thread: bool = False):
    """Create an agent node function for the group chat graph.

    Args:
        agent: The agent instance to wrap.
        use_full_thread: Whether to use full_thread or messages for agent input.
            Only relevant if using GroupChatStateWithFullThread.

    Returns:
        A node function that invokes the agent and updates state.

    """

    def node(state: MessagesState, runtime: Runtime):
        # Get the appropriate message list
        if use_full_thread and hasattr(state, "full_thread"):
            messages = state.get("full_thread", [])
        else:
            messages = state.get("messages", [])

        # Invoke the agent
        response = agent.invoke({"messages": messages}, context=runtime.context)

        # Extract new messages from response
        new_messages = response["messages"]

        # Create return dict
        result = {
            "messages": [
                HumanMessage(content=new_messages[-1].content, name=agent.name)
            ]
        }

        # If state has full_thread, update it too
        if use_full_thread and hasattr(state, "full_thread"):
            result["full_thread"] = new_messages

        return result

    return node


def create_groupchat(
    participants,
    selector_func,
    state_schema: type = GroupChatState,
    context_schema: type[Any] | None = None,
    *,
    use_full_thread: bool = False,
) -> StateGraph:
    """Create a group chat system with dynamic speaker selection.

    This function creates a LangGraph StateGraph that manages multi-agent
    conversations where a selector function dynamically chooses the next
    speaker based on conversation history.

    Args:
        participants: List of agent instances to participate in the group chat.
            Each agent must have a 'name' attribute and an 'invoke' method.
        selector_func: Function that selects the next speaker. Should accept
            state and return the name of the next agent or "Terminate".
        state_schema: Dataclass defining the state schema. Must extend MessagesState
            or have a 'messages' field. Must also have 'roles' field.
            Defaults to GroupChatState.
        context_schema: Optional type definition for runtime context.
        use_full_thread: If True and state_schema has 'full_thread' field,
            agents receive full_thread instead of messages. Defaults to False.

    Returns:
        A StateGraph instance ready to be compiled with a checkpointer.

    Raises:
        ValueError: If state_schema is missing required fields or if
            participants list is empty.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain.agents import create_agent
        >>> from langchain.messages import HumanMessage
        >>>
        >>> model = ChatOpenAI(model="gpt-4o")
        >>>
        >>> # Create agents
        >>> agent1 = create_agent(model, tools=[], name="Alice")
        >>> agent2 = create_agent(model, tools=[], name="Bob")
        >>>
        >>> # Define selector function
        >>> def selector(state):
        ...     # Simple alternating logic
        ...     last_speaker = state.get("messages",[])[-1].name if state.get("messages",[]) else None
        ...     if last_speaker == "Alice":
        ...         return "Bob"
        ...     if last_speaker == "Bob":
        ...         return "Terminate"
        ...     return "Alice"
        >>>
        >>> # Build the group chat using MessagesState
        >>> workflow = create_groupchat(
        ...     participants=[agent1, agent2],
        ...     selector_func=selector,
        ... )
        >>> app = workflow.compile()
        >>>
        >>> # Use the group chat
        >>> response = app.invoke(
        ...     {"messages": [HumanMessage(content="Hi how are you?")]},
        ... )
        >>> print(response['messages'])

    """
    # Validate inputs
    if not participants:
        msg = "participants list cannot be empty"
        raise ValueError(msg)

    agent_names = [agent.name for agent in participants]

    # Validate state schema has required fields
    if "messages" not in state_schema.__annotations__:
        msg = (
            "Missing required key 'messages' in state_schema. "
            "State must extend MessagesState or have a messages field."
        )
        raise ValueError(msg)

    if "roles" not in state_schema.__annotations__:
        msg = (
            "Missing required key 'roles' in state_schema. "
            "This is used to store available participant names."
        )
        raise ValueError(msg)

    # Validate full_thread if use_full_thread is True
    if use_full_thread and "full_thread" not in state_schema.__annotations__:
        msg = (
            "use_full_thread=True but state_schema doesn't have 'full_thread' field. "
            "Use GroupChatStateWithFullThread or add full_thread to your schema."
        )
        raise ValueError(msg)

    # Create updated state schema with agent names in roles
    # This is similar to how swarm updates active_agent to use Literal
    updated_group_chat_state = type(
        f"{state_schema.__name__}_Updated",
        (state_schema,),
        {
            "__annotations__": {**state_schema.__annotations__, "roles": list[str]},
            "roles": field(default_factory=lambda: agent_names.copy()),
        },
    )

    def select_participant(state: updated_group_chat_state):
        """Select the next participant to speak."""
        # Ensure roles are populated for the selector
        if not state.get("roles"):
            # Create a temporary state dict with roles for the selector
            state_with_roles = dict(state)
            state_with_roles["roles"] = agent_names
            speaker = selector_func(state_with_roles)
            update = {"roles": agent_names}  # Persist roles in state
        else:
            speaker = selector_func(state)
            update = {}  # No update needed

        # Validate speaker selection
        if speaker not in agent_names and speaker != "Terminate":
            logger.warning("Invalid speaker '%s', terminating", speaker)
            return Command(goto="__end__", update=update)

        if speaker == "Terminate":
            return Command(goto="__end__", update=update)

        return Command(goto=speaker, update=update)

    # Dynamically create the return type annotation for type safety
    _literal_values = ", ".join([f'"{name}"' for name in [*agent_names, "__end__"]])
    _return_type_str = f"Command[Literal[{_literal_values}]]"
    select_participant.__annotations__["return"] = eval(  # noqa: S307
        _return_type_str, {"Command": Command, "Literal": Literal}
    )

    # Build the graph
    builder = StateGraph(updated_group_chat_state, context_schema)
    builder.add_edge(START, "selector")
    builder.add_node("selector", select_participant)

    # Add agent nodes
    for agent in participants:
        builder.add_node(
            agent.name,
            agent_node(agent, use_full_thread=use_full_thread),
        )
        builder.add_edge(agent.name, "selector")

    return builder
