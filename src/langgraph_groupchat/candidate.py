def no_repeat_candidate(state):
    """Prevent the same speaker from speaking consecutively.

    This candidate function filters out the current speaker from the list of
    available speakers, ensuring that no agent speaks twice in a row. This
    helps create more dynamic and balanced conversations.

    Args:
        state: The current state containing 'messages' and 'roles' fields.
            - messages: List of conversation messages
            - roles: List of all available participant names

    Returns:
        list: Available role names excluding the current speaker. If filtering
        would result in an empty list, returns all roles as a fallback.

    """
    messages = state.get("messages", [])
    roles = state.get("roles", [])

    if not messages:
        return roles

    current_speaker = None
    if messages and hasattr(messages[-1], "name"):
        current_speaker = messages[-1].name

    if current_speaker is None:
        return roles

    available_roles = [role for role in roles if role != current_speaker]

    if not available_roles:
        return roles

    return available_roles
