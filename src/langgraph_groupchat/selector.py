from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from enum import Enum

default_selector_system_prompt = """You are a group chat manager. Your job is to select the next speaker based on the conversation history and the available participants."""

def llm_based_selector(model, system_prompt=default_selector_system_prompt, candidate_func=None):
    speaker_template = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )

    def select_speaker(state):
        messages = state.messages
        if candidate_func is None:
            allowed_roles = state.roles
        else:
            allowed_roles = candidate_func(state)
        
        allowed_roles.append("Terminate")
        
        # Dynamically create an Enum from the allowed roles
        SpeakerEnum = Enum('SpeakerEnum', {role: role for role in allowed_roles})
        
        class SelectParticipant(BaseModel):  
            """Select the next speaker based on the system prompt and messages."""
            next_speaker: SpeakerEnum = Field(description="The next speaker in the conversation")
        
        structured_llm = model.with_structured_output(SelectParticipant)  
        selector_chain = speaker_template | structured_llm
        
        result = selector_chain.invoke({
            "messages": messages
        })
        
        # Return the enum value (string)
        return result.next_speaker.value
    
    return select_speaker