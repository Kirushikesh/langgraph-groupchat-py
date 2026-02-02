def no_repeat_candidate(state):
    """
    Candidate function that prevents the same speaker from speaking consecutively.
    
    Args:
        state: The current state containing 'messages' and 'roles'
        
    Returns:
        list: Available roles excluding the current speaker
    """
    messages = state.messages
    roles = state.roles
    
    # If there are no messages, return all roles
    if not messages:
        return roles
    
    # Get the current speaker from the last message
    current_speaker = None
    if messages and hasattr(messages[-1], 'name'):
        current_speaker = messages[-1].name
    
    # If no current speaker found, return all roles
    if current_speaker is None:
        return roles
    
    # Filter out the current speaker from available roles
    available_roles = [role for role in roles if role != current_speaker]
    
    # If filtering leaves no roles, return all roles (fallback)
    if not available_roles:
        return roles
    
    return available_roles