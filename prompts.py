"""
Forked from https://github.com/nestordemeure/question_extractor
"""
from langchain.schema import HumanMessage, SystemMessage

#----------------------------------------------------------------------------------------
# EXTRACTION

# Improved prompt for extracting questions
extraction_system_prompt = """You are an expert at creating insightful questions from documentation. 
Your task is to generate questions that:
1. Can be fully answered using ONLY the provided text
2. Cover important concepts, details, and information in the text
3. Are specific and targeted rather than overly general
4. Represent the key knowledge someone should gain from this content
5. Have clear, unambiguous answers present in the text

Create a numbered list of questions that meet these criteria."""

def create_extraction_conversation_messages(text):
    """
    Takes a piece of text and returns a list of messages designed to extract questions from the text.
    
    Args:
        text (str): The input text for which questions are to be extracted.
    
    Returns:
        list: A list of messages that set up the context for extracting questions.
    """
    # Create a system message setting the context for the extraction task
    context_message = SystemMessage(content=extraction_system_prompt)
    
    # Create a human message containing the input text
    input_text_message = HumanMessage(content=text)
    
    # Return the list of messages to be used in the extraction conversation
    return [context_message, input_text_message]


#----------------------------------------------------------------------------------------
# ANSWERING

# Improved prompt for answering questions
answering_system_prompt = """You are an expert providing clear, direct answers to questions.
Your answers should:
1. Be based SOLELY on the provided text
2. Begin directly with the answer - do NOT use phrases like "The text states that" or "According to the text"
3. Be comprehensive but concise
4. Use a natural, conversational tone
5. Include relevant details and context from the text
6. If the exact answer is not in the text, respond with "This information is not provided in the text"

Respond in a helpful, straightforward manner without unnecessary preambles."""


def create_answering_conversation_messages(question, text):
    """
    Takes a question and a text and returns a list of messages designed to answer the question based on the text.
    
    Args:
        question (str): The question to be answered.
        text (str): The text containing information for answering the question.
    
    Returns:
        list: A list of messages that set up the context for answering the question.
    """
    # Create a system message setting the context for the answering task
    context_message = SystemMessage(content=answering_system_prompt)
    
    # Create a human message containing the input text
    input_text_message = HumanMessage(content=text)
    
    # Create a human message containing the question to be answered
    input_question_message = HumanMessage(content=question)
    
    # Return the list of messages to be used in the answering conversation
    return [context_message, input_text_message, input_question_message]
