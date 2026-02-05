# 🤖 LangGraph Group Chat

A Python library for creating group chat-style multi-agent systems using [LangGraph](https://github.com/langchain-ai/langgraph). A group chat is a type of [multi-agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent) architecture where a **selector function** dynamically chooses the next speaker based on conversation history and context. Unlike swarm patterns where agents hand off control to each other, group chat uses centralized orchestration to manage turn-taking among multiple specialized agents.

![Group Chat Pattern](static/img/groupchat.svg)

## Features

- 🎯 **Dynamic speaker selection** - Intelligent routing based on conversation context using LLM-based or custom selectors
- 🔄 **Flexible conversation flow** - Support for rule-based, LLM-based, or hybrid speaker selection strategies  
- 🛠️ **Candidate filtering** - Control which agents can speak next based on custom logic (e.g., prevent consecutive turns)
- 📊 **Dual message tracking** - Optional separate tracking of full conversation thread vs. filtered messages for agents
- 🎨 **Customizable state schemas** - Extend the base state with custom fields for your specific use case

This library is built on top of [LangGraph](https://github.com/langchain-ai/langgraph), a powerful framework for building agent applications, and comes with out-of-box support for [streaming](https://langchain-ai.github.io/langgraph/how-tos/#streaming), [short-term and long-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/), and [human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).

## When to Use Group Chat

Group chat is ideal when you need:

- **Centralized orchestration** - A coordinator that decides who speaks next based on the overall conversation state
- **Complex turn-taking logic** - Rules about speaker selection that go beyond simple handoffs
- **Moderated discussions** - Preventing agents from monopolizing the conversation or speaking out of turn
- **Dynamic participation** - Context-aware agent selection where the best-suited agent speaks next
- **Broadcast communication** - All agents share the same conversation context (like a team meeting)

## Installation

```bash
pip install langgraph-groupchat
```

## Quickstart

```bash
pip install langgraph-groupchat langchain-openai

export OPENAI_API_KEY=<your_api_key>
```

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_groupchat import create_groupchat, llm_based_selector

model = ChatOpenAI(model="gpt-4o")

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# Create specialized agents
alice = create_agent(
    model,
    tools=[add],
    system_prompt="You are Alice, an addition expert. You excel at adding numbers.",
    name="Alice",
)

bob = create_agent(
    model,
    tools=[multiply],
    system_prompt="You are Bob, a multiplication expert. You excel at multiplying numbers.",
    name="Bob",
)

charlie = create_agent(
    model,
    tools=[],
    system_prompt="You are Charlie, a helpful assistant who coordinates math tasks.",
    name="Charlie",
)

# Create an LLM-based selector that chooses the next speaker
selector = llm_based_selector(model=model)

# Build the group chat
checkpointer = InMemorySaver()
workflow = create_groupchat(
    participants=[alice, bob, charlie],
    selector_func=selector,
)
app = workflow.compile(checkpointer=checkpointer)

# Use the group chat
config = {"configurable": {"thread_id": "1"}}
result = app.invoke(
    {"messages": [{"role": "user", "content": "What is 5 + 7 and then multiply that by 3?"}]},
    config,
)
print(result["messages"])
```

## Core Concepts

### Selector Functions

The selector function is the heart of the group chat pattern. It examines the conversation state and returns the name of the next speaker or `"Terminate"` to end the conversation.

#### LLM-Based Selector

Use an LLM to intelligently choose the next speaker:

```python
from langgraph_groupchat import llm_based_selector

# Basic LLM selector
selector = llm_based_selector(model=model)

# With custom system prompt
selector = llm_based_selector(
    model=model,
    system_prompt="""You are managing a technical support chat.
    Select the specialist best suited to handle the current issue.
    - Alice handles billing questions
    - Bob handles technical issues  
    - Charlie handles general inquiries"""
)
```

#### Custom Rule-Based Selector

Create your own selector logic:

```python
def round_robin_selector(state):
    """Simple round-robin speaker selection."""
    messages = state.get("messages", [])
    roles = state.get("roles", [])
    
    if not messages:
        return roles[0]
    
    # Get current speaker
    last_speaker = messages[-1].name if hasattr(messages[-1], "name") else None
    
    # Find next speaker in round-robin order
    if last_speaker and last_speaker in roles:
        current_idx = roles.index(last_speaker)
        next_idx = (current_idx + 1) % len(roles)
        return roles[next_idx]
    
    return roles[0]

workflow = create_groupchat(
    participants=[alice, bob, charlie],
    selector_func=round_robin_selector,
)
```

#### Conditional Selector

Implement complex routing logic:

```python
def conditional_selector(state):
    """Route based on conversation content and state."""
    messages = state.get("messages", [])
    
    if not messages:
        return "Charlie"  # Start with coordinator
    
    last_message = messages[-1].content.lower()
    
    # Route based on keywords
    if "add" in last_message or "sum" in last_message:
        return "Alice"
    elif "multiply" in last_message or "times" in last_message:
        return "Bob"
    elif "done" in last_message or "finished" in last_message:
        return "Terminate"
    else:
        return "Charlie"  # Default to coordinator
```

### Candidate Functions

Candidate functions filter which agents can be selected. This is useful for preventing consecutive turns, implementing speaking order constraints, or dynamically adjusting available speakers.

#### No Repeat Candidate

Prevent the same agent from speaking twice in a row:

```python
from langgraph_groupchat import llm_based_selector, no_repeat_candidate

selector = llm_based_selector(
    model=model,
    candidate_func=no_repeat_candidate
)
```

#### Custom Candidate Function

Create your own filtering logic:

```python
def domain_based_candidate(state):
    """Only allow agents relevant to the current domain."""
    messages = state.get("messages", [])
    roles = state.get("roles", [])
    
    if not messages:
        return roles
    
    last_message = messages[-1].content.lower()
    
    # Filter candidates based on domain
    if "math" in last_message or "calculate" in last_message:
        return [r for r in roles if r in ["Alice", "Bob"]]
    else:
        return roles

selector = llm_based_selector(
    model=model,
    candidate_func=domain_based_candidate
)
```

### State Schemas

Group chat supports two state schemas:

#### GroupChatState (Default)

The basic state schema with shared message history:

```python
from langgraph_groupchat import GroupChatState, create_groupchat

workflow = create_groupchat(
    participants=[alice, bob, charlie],
    selector_func=selector,
    state_schema=GroupChatState,  # Default
)
```

#### GroupChatStateWithFullThread

For advanced use cases where you need separate message tracking:

```python
from langgraph_groupchat import GroupChatStateWithFullThread, create_groupchat

workflow = create_groupchat(
    participants=[alice, bob, charlie],
    selector_func=selector,
    state_schema=GroupChatStateWithFullThread,
    use_full_thread=True,  # Agents receive full_thread instead of messages
)
```

This allows you to:
- Keep internal agent reasoning separate from the main conversation
- Filter what users see vs. what agents process
- Maintain different message histories for different purposes

## Memory

You can add [short-term](https://langchain-ai.github.io/langgraph/how-tos/persistence/) and [long-term](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/) [memory](https://langchain-ai.github.io/langgraph/concepts/memory/) to your group chat system. Since `create_groupchat()` returns an instance of `StateGraph` that needs to be compiled before use, you can directly pass a [checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver) or a [store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) instance to the `.compile()` method:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# short-term memory
checkpointer = InMemorySaver()
# long-term memory
store = InMemoryStore()

workflow = create_groupchat(
    participants=[alice, bob, charlie],
    selector_func=selector,
)

# Compile with checkpointer/store
app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)
```

> [!IMPORTANT]
> Adding [short-term memory](https://langchain-ai.github.io/langgraph/concepts/persistence/) is crucial for maintaining conversation state across multiple interactions. Without it, the group chat would lose the conversation history between invocations. Make sure to always compile the group chat with a checkpointer if you plan to use it in multi-turn conversations; e.g., `workflow.compile(checkpointer=checkpointer)`.

## Advanced Examples

### Multi-Domain Expert System

Create a group chat with domain-specific experts and an intelligent coordinator:

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph_groupchat import create_groupchat, llm_based_selector, no_repeat_candidate

model = ChatOpenAI(model="gpt-4o")

# Domain experts
python_expert = create_agent(
    model,
    tools=[],
    system_prompt="""You are a Python programming expert. 
    Help users with Python code, best practices, and debugging.
    When you're done, say 'Python expertise complete.'""",
    name="PythonExpert",
)

database_expert = create_agent(
    model,
    tools=[],
    system_prompt="""You are a database expert specializing in SQL and database design.
    Help users with queries, optimization, and schema design.
    When you're done, say 'Database expertise complete.'""",
    name="DatabaseExpert",
)

coordinator = create_agent(
    model,
    tools=[],
    system_prompt="""You are a technical coordinator. 
    Assess user questions and delegate to the appropriate expert.
    Synthesize expert responses into clear answers.
    When the task is complete, say 'TERMINATE'.""",
    name="Coordinator",
)

# Create selector with intelligent routing
custom_prompt = """You are managing a technical consultation.
Based on the conversation history, select the next speaker:
- Coordinator: For initial assessment and final synthesis
- PythonExpert: When Python programming help is needed
- DatabaseExpert: When database help is needed
- Terminate: When the user's question is fully answered"""

selector = llm_based_selector(
    model=model,
    system_prompt=custom_prompt,
    candidate_func=no_repeat_candidate,
)

workflow = create_groupchat(
    participants=[coordinator, python_expert, database_expert],
    selector_func=selector,
)

app = workflow.compile(checkpointer=InMemorySaver())
```

### Research Team with Planned Workflow

Combine rule-based and LLM-based selection for structured research tasks:

```python
def research_selector(state):
    """Structured workflow for research tasks."""
    messages = state.get("messages", [])
    roles = state.get("roles", [])
    
    if not messages:
        return "Planner"
    
    last_speaker = messages[-1].name if hasattr(messages[-1], "name") else None
    last_content = messages[-1].content.lower()
    
    # Structured workflow
    if last_speaker == "Planner":
        # After planning, start research
        return "Researcher"
    elif last_speaker == "Researcher":
        # After research, analyze data
        return "Analyst"
    elif last_speaker == "Analyst":
        # After analysis, write report
        return "Writer"
    elif last_speaker == "Writer":
        # After writing, review
        return "Reviewer"
    elif last_speaker == "Reviewer":
        if "approved" in last_content:
            return "Terminate"
        elif "revision needed" in last_content:
            return "Writer"
        else:
            return "Terminate"
    
    return "Planner"

planner = create_agent(model, tools=[], name="Planner", 
    system_prompt="Break down research tasks into clear steps.")
researcher = create_agent(model, tools=[], name="Researcher",
    system_prompt="Gather information on the research topic.")
analyst = create_agent(model, tools=[], name="Analyst",
    system_prompt="Analyze research findings and identify key insights.")
writer = create_agent(model, tools=[], name="Writer",
    system_prompt="Write a clear, well-structured report.")
reviewer = create_agent(model, tools=[], name="Reviewer",
    system_prompt="Review the report. Say 'Approved' or 'Revision needed: <feedback>'.")

workflow = create_groupchat(
    participants=[planner, researcher, analyst, writer, reviewer],
    selector_func=research_selector,
)
```

## Comparing Group Chat with Other Multi-Agent Patterns

Group chat is one of several multi-agent patterns available in LangGraph. Here's how it compares:

### Pattern Overview

| Pattern | Coordination | Speaker Selection | Best For |
|---------|-------------|-------------------|----------|
| **Group Chat** | Centralized (Selector) | Dynamic (LLM or rules) | Complex discussions, moderated collaboration |
| **[Swarm](https://github.com/langchain-ai/langgraph-swarm)** | Decentralized (Handoffs) | Agent-driven (via tools) | Fluid handoffs, direct user interaction |
| **[Router](https://docs.langchain.com/oss/python/langchain/multi-agent/router)** | Centralized (Router) | Upfront classification | Parallel queries, distinct domains |
| **[Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)** | Hierarchical (Main agent) | Main agent decides | Supervisor-worker, tool delegation |
| **[Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)** | Decentralized | State-driven tools | Context transfers, agent specialization |

### Key Differences

#### Group Chat vs. Swarm

**Group Chat:**
- ✅ Centralized speaker selection via selector function
- ✅ Sequential turns with a single active speaker
- ✅ Selector can prevent consecutive turns or enforce speaking order
- ✅ All agents see the full conversation history (broadcast model)
- ❌ Less agent autonomy (selector controls flow)
- ❌ Cannot hand off mid-conversation (selector decides next)

**Swarm:**
- ✅ Agents hand off via tools (decentralized control)
- ✅ Remembers last active agent for seamless continuation
- ✅ More agent autonomy (agents decide when to transfer)
- ✅ Direct agent-to-agent handoffs
- ❌ No central orchestration of turn-taking
- ❌ Less control over speaking order

**When to choose:**
- Use **Group Chat** when you need centralized moderation, complex turn-taking rules, or want to prevent certain speaking patterns
- Use **Swarm** when agents should control their own handoffs and you want fluid, autonomous transfers

#### Group Chat vs. Router

**Group Chat:**
- Sequential turns with conversation history
- Dynamic selection at each turn based on full context
- Supports multi-turn interactions between agents
- Best for: Collaborative discussions, iterative problem-solving

**Router:**
- Upfront classification of the request
- Parallel execution (can query multiple agents at once)
- Results synthesized into final response
- Best for: One-shot queries, multi-domain searches, parallel data gathering

**When to choose:**
- Use **Group Chat** for multi-turn conversations where agents build on each other's contributions
- Use **Router** for parallelizable queries across distinct knowledge domains

#### Group Chat vs. Subagents

**Group Chat:**
- All participants are peers
- Selector function orchestrates (separate from agents)
- Broadcast communication model
- Best for: Team discussions, peer collaboration

**Subagents:**
- Hierarchical structure (main agent + subagents)
- Main agent orchestrates via tool calls
- Parent-child relationship
- Best for: Supervisor-worker patterns, delegation

**When to choose:**
- Use **Group Chat** when all agents should have equal status and share context
- Use **Subagents** when you need a clear supervisor coordinating specialized workers

### AutoGen Comparison

This implementation is inspired by [AutoGen's SelectorGroupChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html):

**Similarities:**
- Both use a selector to choose the next speaker
- Support LLM-based and custom selector functions
- Allow candidate filtering (e.g., prevent consecutive turns)
- All participants share conversation context

**Key Differences:**

| Feature | LangGraph Group Chat | AutoGen SelectorGroupChat |
|---------|---------------------|---------------------------|
| Framework | LangGraph (event-driven) | AutoGen (message-based) |
| Selector | Separate function | Integrated into GroupChatManager |
| State Management | LangGraph state with checkpointers | AutoGen's built-in state |
| Customization | Custom state schemas, dual-thread support | Custom speaker selection methods |
| Integration | Works with LangGraph ecosystem | Works with AutoGen ecosystem |

### Performance Characteristics

Here's how group chat performs compared to other patterns:

**Scenario: Multi-turn conversation (3 agents, 5 turns)**

| Pattern | Model Calls | Tokens Processed | Parallelization |
|---------|-------------|------------------|-----------------|
| Group Chat | 6 (1 per turn + selector) | ~15K | Sequential |
| Swarm | 5 (no selector overhead) | ~12K | Sequential |
| Router (stateless) | 15 (3 per request) | ~10K | Parallel |
| Subagents | 10 (2 per turn) | ~14K | Can parallelize |

**Tradeoffs:**
- Group Chat adds selector overhead but provides flexible orchestration
- Best for: Complex turn-taking where selector logic adds value
- Consider alternatives if: Selector adds latency without benefit (use Swarm) or parallelization is key (use Router)

## How to Customize

### Customizing the Selector

#### Hybrid Selectors

Combine rules and LLM intelligence:

```python
def hybrid_selector(state):
    """Use rules when possible, LLM when needed."""
    messages = state.get("messages", [])
    roles = state.get("roles", [])
    
    if not messages:
        return "Coordinator"
    
    last_message = messages[-1].content.lower()
    
    # Simple rules for common patterns
    if "terminate" in last_message or "done" in last_message:
        return "Terminate"
    
    # Fallback to LLM for complex decisions
    llm_selector = llm_based_selector(model=model)
    return llm_selector(state)
```

#### Context-Aware Selectors

Access full state for intelligent routing:

```python
def context_aware_selector(state):
    """Make selection based on conversation context."""
    messages = state.get("messages", [])
    roles = state.get("roles", [])
    
    # Count how many times each agent has spoken
    speaker_counts = {}
    for msg in messages:
        if hasattr(msg, "name") and msg.name:
            speaker_counts[msg.name] = speaker_counts.get(msg.name, 0) + 1
    
    # Balance participation - prefer agents who have spoken less
    available = [r for r in roles if speaker_counts.get(r, 0) < 3]
    
    if not available:
        return "Terminate"
    
    # Use LLM to select from balanced candidates
    selector = llm_based_selector(model=model, candidate_func=lambda s: available)
    return selector(state)
```

### Customizing Agent Implementation

By default, agents communicate over a shared `messages` key. For more control, you can customize how agents interact with state:

#### Custom State Schema

```python
from dataclasses import dataclass, field
from typing import Annotated
from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph_groupchat import GroupChatState

@dataclass
class CustomGroupChatState(GroupChatState):
    """Extended state with additional fields."""
    task_description: str = ""
    completed_subtasks: list[str] = field(default_factory=list)
    current_domain: str = ""

workflow = create_groupchat(
    participants=[alice, bob, charlie],
    selector_func=selector,
    state_schema=CustomGroupChatState,
)
```

#### Separate Message Histories

Keep agent internal reasoning separate from user-facing messages:

```python
from langgraph_groupchat import GroupChatStateWithFullThread

# Agents see full_thread (including internal reasoning)
# Users see messages (filtered conversation)
workflow = create_groupchat(
    participants=[alice, bob, charlie],
    selector_func=selector,
    state_schema=GroupChatStateWithFullThread,
    use_full_thread=True,
)
```

### Termination Conditions

Control when the conversation ends:

#### Keyword-Based Termination

```python
def terminating_selector(state):
    """Terminate when specific keywords are detected."""
    messages = state.get("messages", [])
    
    if messages and any(
        keyword in messages[-1].content.lower() 
        for keyword in ["finished", "complete", "terminate", "done"]
    ):
        return "Terminate"
    
    # Otherwise, use normal selection
    return llm_based_selector(model=model)(state)
```

#### Max Turns Termination

```python
def max_turns_selector(state, max_turns=10):
    """Terminate after maximum number of turns."""
    messages = state.get("messages", [])
    
    if len(messages) >= max_turns:
        return "Terminate"
    
    return llm_based_selector(model=model)(state)
```

#### Consensus-Based Termination

```python
def consensus_selector(state):
    """Terminate when all agents agree task is complete."""
    messages = state.get("messages", [])
    roles = state.get("roles", [])
    
    # Check recent messages for completion signals
    recent_messages = messages[-len(roles):] if len(messages) >= len(roles) else messages
    
    completion_keywords = ["complete", "finished", "done"]
    agents_done = set()
    
    for msg in recent_messages:
        if hasattr(msg, "name") and any(kw in msg.content.lower() for kw in completion_keywords):
            agents_done.add(msg.name)
    
    # If all agents have signaled completion, terminate
    if len(agents_done) >= len(roles):
        return "Terminate"
    
    return llm_based_selector(model=model)(state)
```

## Best Practices

1. **Design clear agent roles**: Each agent should have a well-defined specialty and clear system prompt
2. **Use candidate functions wisely**: Prevent unproductive patterns (e.g., consecutive turns) while allowing flexibility
3. **Choose the right selector**: Use rules when logic is clear, LLM when you need intelligence, hybrid for best of both
4. **Add memory**: Always use a checkpointer for multi-turn conversations
5. **Monitor token usage**: Group chats share full history - consider message summarization for long conversations
6. **Test termination**: Ensure conversations can end gracefully without getting stuck
7. **Balance control vs. autonomy**: Too much selector control can feel rigid; too little can be chaotic

## Examples and Tutorials

Check out the `examples/` directory for complete implementations:

- `basic_groupchat.py` - Simple group chat with three agents
- `research_team.py` - Structured research workflow
- `customer_support.py` - Multi-domain customer service
- `code_review.py` - Collaborative code review system

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [AutoGen's SelectorGroupChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html)
- Built on [LangGraph](https://github.com/langchain-ai/langgraph)
- Pattern comparisons reference [LangChain's multi-agent documentation](https://docs.langchain.com/oss/python/langchain/multi-agent)