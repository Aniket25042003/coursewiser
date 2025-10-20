"""
Inference service with RAG prompt building and response cleaning
"""
import re
from typing import List, Dict, Optional
from app.services.model_loader import get_model_wrapper


# System prompt for DSA tutoring
SYSTEM_PROMPT = """You are an expert Data Structures and Algorithms (DSA) tutor. Your role is to:
- Explain DSA concepts clearly and concisely
- Provide accurate code examples when relevant
- Help students understand algorithms and their time/space complexity
- Answer only DSA-related questions
- Refuse to answer questions unrelated to DSA, programming, or computer science
- Be encouraging and supportive to students

If asked about harmful, unethical, or non-DSA topics, politely decline and redirect to DSA topics."""


# Safety keywords to check
HARMFUL_KEYWORDS = [
    "bomb", "weapon", "kill", "murder", "suicide", "hack", "steal", 
    "illegal", "drug", "violence", "racist", "sexist"
]


def check_safety(text: str) -> bool:
    """
    Check if text contains harmful keywords
    
    Returns:
        True if safe, False if potentially harmful
    """
    text_lower = text.lower()
    for keyword in HARMFUL_KEYWORDS:
        if keyword in text_lower:
            return False
    return True


def clean_response(raw_output: str, original_prompt: str) -> str:
    """
    Clean model output by removing prompt echo and excessive repetition
    
    Args:
        raw_output: Raw model output
        original_prompt: The prompt that was sent to the model
        
    Returns:
        Cleaned response text
    """
    # Remove the prompt from the output
    if original_prompt in raw_output:
        response = raw_output.replace(original_prompt, "").strip()
    else:
        # Try to extract everything after "### Response:"
        if "### Response:" in raw_output:
            response = raw_output.split("### Response:")[-1].strip()
        else:
            response = raw_output
    
    # Remove excessive newlines
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Remove repetitive patterns (simple check)
    lines = response.split('\n')
    unique_lines = []
    for line in lines:
        if not unique_lines or line != unique_lines[-1]:
            unique_lines.append(line)
    response = '\n'.join(unique_lines)
    
    # Safety check on output
    if not check_safety(response):
        return "I apologize, but I can only help with Data Structures and Algorithms questions. Please ask a DSA-related question."
    
    return response.strip()


def build_rag_prompt(
    user_question: str,
    retrieved_chunks: Optional[List[Dict]] = None,
    conversation_history: Optional[List[Dict]] = None,
    system_prompt: str = SYSTEM_PROMPT
) -> str:
    """
    Build a complete RAG prompt with system instructions, context, history, and user question
    
    Args:
        user_question: The user's current question
        retrieved_chunks: List of retrieved context chunks with 'text' and 'metadata'
        conversation_history: List of previous chat turns with 'message' and 'response'
        system_prompt: System instructions
        
    Returns:
        Complete formatted prompt
    """
    prompt = f"### System:\n{system_prompt}\n\n"
    
    # Add retrieved context
    if retrieved_chunks and len(retrieved_chunks) > 0:
        prompt += "### Context:\n"
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk.get('text', chunk.get('chunk_text', ''))
            prompt += f"{chunk_text}\n---\n"
        prompt += "\n"
    
    # Add conversation history (last 4 turns to keep context manageable)
    if conversation_history and len(conversation_history) > 0:
        prompt += "### Conversation:\n"
        for turn in conversation_history[-4:]:
            prompt += f"User: {turn['message']}\n"
            prompt += f"Assistant: {turn['response']}\n"
        prompt += "\n"
    
    # Add current instruction
    prompt += f"### Instruction:\n{user_question}\n\n### Response:\n"
    
    return prompt


def generate_response_with_rag(
    user_question: str,
    retrieved_chunks: Optional[List[Dict]] = None,
    conversation_history: Optional[List[Dict]] = None,
    max_new_tokens: int = 200
) -> str:
    """
    Generate a response using RAG with the fine-tuned model
    
    Args:
        user_question: The user's question
        retrieved_chunks: Retrieved context from vector database
        conversation_history: Previous conversation turns
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated and cleaned response
    """
    # Safety pre-check
    if not check_safety(user_question):
        return "I apologize, but I can only help with Data Structures and Algorithms questions. Please ask a DSA-related question."
    
    # Build prompt
    prompt = build_rag_prompt(user_question, retrieved_chunks, conversation_history)
    
    # Generate response
    model = get_model_wrapper()
    raw_output = model.generate_from_prompt(prompt, max_new_tokens=max_new_tokens)
    
    # Clean and return
    cleaned = clean_response(raw_output, prompt)
    
    return cleaned

