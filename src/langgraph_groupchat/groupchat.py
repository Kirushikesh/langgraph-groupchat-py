from typing import List, Literal, Any, Union
from langgraph.runtime import Runtime

from langchain_core.messages import HumanMessage
from langchain.messages import AnyMessage
from typing_extensions import Annotated
import operator
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command
from langgraph.graph import START
from dataclasses import dataclass, field, fields, make_dataclass

# Base state schema that users can use or extend
@dataclass
class GroupChatState:
    """State schema for group chat."""
    messages: Annotated[list[AnyMessage], operator.add] = field(default_factory=list)
    full_thread: Annotated[list[AnyMessage], operator.add] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)


def update_state_with_roles(
    state_schema: type,
    agent_names: list[str]
) -> type:
    """
    Create a new dataclass with updated default values for the 'roles' field.
    
    Args:
        state_schema: The original dataclass state schema
        agent_names: List of agent names to use as default for roles
        
    Returns:
        A new dataclass with updated roles default
    """
    # Get all fields from the original dataclass
    original_fields = fields(state_schema)
    
    # Get type hints to preserve Annotated types
    type_hints = get_type_hints(state_schema, include_extras=True)
    
    # Build new field definitions
    new_fields = []
    for f in original_fields:
        field_name = f.name
        field_type = type_hints.get(field_name, f.type)
        
        if field_name == "roles":
            # Update the roles field with new default
            new_fields.append((
                field_name,
                field_type,
                field(default_factory=lambda: agent_names.copy())
            ))
        else:
            # Keep other fields as-is
            if f.default is not f.default_factory:
                # Has a default value
                new_fields.append((field_name, field_type, f.default))
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore
                # Has a default_factory
                new_fields.append((field_name, field_type, field(default_factory=f.default_factory)))
            else:
                # No default
                new_fields.append((field_name, field_type))
    
    # Create new dataclass with same name
    return make_dataclass(
        state_schema.__name__,
        new_fields,
        bases=(state_schema.__bases__ if hasattr(state_schema, '__bases__') else ()),
    )

def agent_node(agent, use_full_thread, updated_state_schema):
    def node(state: updated_state_schema, runtime: Runtime):
        messages = state.full_thread if use_full_thread else state.messages
        response = agent.invoke({"messages": messages}, context=runtime.context)

        return {
            "full_thread": state.full_thread + response['messages'],
            "messages": state.messages + [HumanMessage(content=response['messages'][-1].content, name=agent.name)]
        }
    return node

def create_groupchat(
    participants,
    selector_func,
    state_schema = GroupChatState,
    context_schema: type[Any] | None = None,
    use_full_thread: bool = False
) -> StateGraph:

    # Get agent names for default roles  
    agent_names = [agent.name for agent in participants]  

    thread_annotation = state_schema.__annotations__.get("full_thread")
    if thread_annotation is None:
        msg = "Missing required key 'full_thread' in state_schema. Which is used to store the full conversation history."
        raise ValueError(msg)
    
    if not state_schema.__annotations__.get("messages"):
        msg = "Missing required key 'messages' in state_schema. Which is used to store the conversation history."
        raise ValueError(msg)
    
    if not participants:
        msg = "participants list cannot be empty"
        raise ValueError(msg)
    
    if not state_schema.__annotations__.get("roles"):
        msg = "Missing required key 'roles' in state_schema. Which is used to store the conversation history."
        raise ValueError(msg)
    
    # Update state schema with agent names as default roles
    # UpdatedGroupChatState = update_state_with_roles(state_schema, agent_names)
    @dataclass
    class UpdatedGroupChatState:
        messages: Annotated[list[AnyMessage], operator.add] = field(default_factory=list)
        full_thread: Annotated[list[AnyMessage], operator.add] = field(default_factory=list)
        roles: list[str] = field(default_factory=lambda: agent_names.copy())

    # Define the function without type hint first
    def select_participant(state: UpdatedGroupChatState):
        speaker = selector_func(state) 
        if(speaker == "Terminate"):
            return Command(
                goto="__end__"
            )
        else:
            return Command(
                goto=speaker
            )

    # Dynamically create and set the return type annotation
    _literal_values = ", ".join([f'"{name}"' for name in agent_names + ["__end__"]])
    _return_type_str = f"Command[Literal[{_literal_values}]]"

    # Set the annotation (this is what LangGraph reads for visualization)
    select_participant.__annotations__['return'] = eval(_return_type_str, {
        'Command': Command,
        'Literal': Literal
    })

    builder = StateGraph(UpdatedGroupChatState, context_schema)
    builder.add_edge(START, "selector")
    builder.add_node("selector", select_participant)
    
    for agent in participants:
        builder.add_node(
            agent.name,
            agent_node(agent, use_full_thread, UpdatedGroupChatState),
        )
        builder.add_edge(agent.name, "selector")

    return builder